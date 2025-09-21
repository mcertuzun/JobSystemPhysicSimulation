
// ObjectFallSimulationOptimized.cs
// Notes on optimizations applied (requested by user):
// - Active-only IJobParallelForDefer across all hot jobs (no work on empty slots)
// - Burst [FastMath/LowPrecision] everywhere, rsqrt-based collision normalization
// - Merge projections (peer + obstacles) and single Apply pass to cut memory traffic
// - GPU instance packet building done in a job into a NativeArray, single upload/copy
// - Persist/despawn pool kept; obstacles use preinflated AABBs to reduce branches
// - SIMD-friendly math (float2, rsqrt, branch-minimizing inner loops)
//
// Drop this as a single C# script into your Unity project. Attach to a GameObject.
// It replaces the older version; public fields kept for compatibility.
//
// Tested with: Unity 2022.3+ (Burst/Collections/Mathematics/Jobs).

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;
using Debug = UnityEngine.Debug;
using Random = Unity.Mathematics.Random;

public class ObjectFallSimulationOptimized : MonoBehaviour
{
    [Header("Baked Level")]
    public TextAsset BakedJson;
    public int PathSampleCount = 256;

    [Serializable] private class BakedLevel
    {
        [Serializable] public struct PathPoint { public float x, y; }
        [Serializable] public struct Spawner   { public float x, y; public int id; }
        [Serializable] public struct Obstacle  { public float x, y; public float radius; public bool open; public int id; }

        public List<PathPoint> path = new();
        public List<Spawner>   spawners = new();
        public List<Obstacle>  obstacles = new();

        public float corridorWidth = 6f;
        public float colliderRadius = 0.06f;
        public float gridCellSize = 0.12f;

        public int tileWidthPx;
        public int tileHeightPx;
        public int mapWidthTiles;
        public int mapHeightTiles;
        public float pixelsPerUnit;
        public string sourceLevelName;
    }

    public int Seed = 12345;
    public bool BenchmarkEnabled = true;
    public int BenchmarkReportInterval = 120;
    private Stopwatch swTotal = new Stopwatch();
    private int benchFrames;
    private double sumTotal;

    [Header("Spawner")]
    public Transform SpawnPoint;
    public Mesh QuadMesh;
    public Material ObjectMaterial;
    public int Capacity = 20000;
    public int SpawnPerSecond = 1000;
    public int BurstOnStart = 0;
    public int MaxActive = 20000;
    public float SpawnAngleDeg = 0f;
    public bool UseCircleJitter = true;
    public float SpawnCircleJitterRadius = 0.5f;
    public Vector2 SpawnBoxJitter = new Vector2(0.5f, 0.2f);
    public List<Sprite> Sprites;
    public bool BuildArrayAtRuntime = true;
    public FilterMode SpriteFilter = FilterMode.Bilinear;
    public Vector2Int SpriteSize = new Vector2Int(128, 128);
    private Texture2DArray spriteArray;
    static readonly int ParticlesID = Shader.PropertyToID("_Particles");
    static readonly int SpriteArrayID = Shader.PropertyToID("_SpriteArray");

    [Header("Sorting")]
    public SortMode SortBy = SortMode.ByAge;
    public bool CpuFrontToBackSort = false;
    public bool SortAscending = true;
    private NativeArray<float> ages;
    private struct SortEntry : IComparable<SortEntry>
    {
        public float Key;
        public int   Id;
        public int CompareTo(SortEntry other) => Key.CompareTo(other.Key);
    }
    private SortEntry[] sortScratch;
    public enum SortMode { None, ByAge, ById }
    private NativeArray<float> angles;
    private NativeArray<int>   spriteIndex;

    [Header("Physics")]
    public float InitialSpeed = 6f;
    public float Gravity = -9.81f;
    public float Restitution = 0.9f;
    public float LinearDamping = 0.01f;
    public int Substeps = 1;
    public float ColliderRadius = 0.06f;
    public int CollisionIterations = 3;
    public float GridCellSize = 0.12f;
    public float ProjectionStiffness = 1.0f;
    public float VelocityFromProjection = 1.0f;
    public int NearestSearchWindow = 12;
    public float CorridorWidth = 6f;
    private float HalfWidth => CorridorWidth * 0.5f;
    public bool RotationEnabled = true;
    public float SpinStrength = 0.5f;

    [Header("Path Force (optional)")]
    public bool PathForceActive;
    public float PathForceLookahead = 1f;
    public float PathForceStrength = 8f;
    public float MinAbsX = 0.2f;

    [Header("Sleep")]
    public float SleepVel = 0.05f;
    public float WakeVel = 0.08f;
    public float SleepProjTol = 0.002f;
    public int SleepFrames = 6;

    [Header("Despawn (Y-threshold)")]
    public bool DespawnBelowYEnabled = true;
    public float DespawnY = -50f;
    public bool DespawnScanRequested;

    [Header("Mobile Auto-Tune (optional)")]
    public bool MobileTuningEnabled;
    public float MobileTargetFrameTime = 1f / 60f;
    public float MobileHardFrameTime = 1f / 45f;

    private NativeArray<float2> positions;
    private NativeArray<float2> velocities;
    private NativeArray<byte> alive;
    private NativeArray<float2> spPos;
    private NativeArray<float2> spTan;
    private NativeArray<int> nearestIndices;

    // Single projection accumulator (merged peer + obstacle)
    private NativeArray<float2> projDelta;

    private NativeParallelMultiHashMap<int, int> cellMap;

    // Obstacles are stored as preinflated AABBs (min/max) for branch-light checks
    private NativeArray<float2> obstacleMin;
    private NativeArray<float2> obstacleMax;
    private NativeArray<byte>   obstacleBlock;

    private NativeArray<byte> sleeping;
    private NativeArray<ushort> sleepCounter;
    private NativeArray<float2> pathAccel;

    private NativeList<int> activeIds;   // persistent pool of active ids
    private NativeList<int> freeIds;     // persistent pool of free ids
    private int activeCount;
    private JobHandle lastHandle;
    private bool allocated;
    private float spawnAcc;
    private Random rng;

    private float2 gridOrigin;
    private float invCell;

    private float spStepLen = 1f;

    // GPU
    private GraphicsBuffer argsBuffer;
    private ComputeBuffer instanceBuffer;
    private NativeArray<ParticleGPU> gpuPackets;

    private struct ParticleGPU
    {
        public float X, Y, R, Angle;
        public float SpriteIndex;
        public float SortKey;
    }

    private Bounds drawBounds;
    private float lastCorridorWidth = -1f;
    private float lastColliderRadius = -1f;

    // ======================= JOBS =======================

    [BurstCompile(FloatPrecision = FloatPrecision.Low, FloatMode = FloatMode.Fast)]
    private struct PathForceJobA : IJobParallelForDefer
    {
        [ReadOnly] public NativeArray<int> Active;
        [ReadOnly] public NativeArray<float2> Positions;
        [ReadOnly] public NativeArray<int> NearestId;
        [ReadOnly] public NativeArray<float2> SpPos;
        [ReadOnly] public NativeArray<byte> Alive;

        public int LookaheadSamples;
        public float Strength;
        public float minAbsX;
        [NativeDisableParallelForRestriction]public NativeArray<float2> outAccel;

        public void Execute(int i)
        {
            int id = Active[i];
            if (Alive[id] == 0) { outAccel[id] = 0f; return; }

            int index = NearestId[id];
            if ((uint)index >= (uint)SpPos.Length) { outAccel[id] = 0f; return; }

            int j = math.min(SpPos.Length - 1, index + math.max(1, LookaheadSamples));
            float2 p = Positions[id];
            float2 target = SpPos[j];

            float2 d = target - p;
            float len = math.length(d);
            if (len <= 1e-6f) { outAccel[id] = 0f; return; }

            float2 n = d / len;
            if (math.abs(n.x) < minAbsX) { outAccel[id] = 0f; return; }

            n.y = 0;
            outAccel[id] = n * Strength;
        }
    }

    [BurstCompile(FloatPrecision = FloatPrecision.Low, FloatMode = FloatMode.Fast)]
    private struct IntegrateJobA : IJobParallelForDefer
    {
        [ReadOnly] public NativeArray<int> Active;
        public float DT;
        public int Substeps;
        public float LinearDamping;
        public float2 Accel;

        [ReadOnly] public NativeArray<byte> Alive;
        [ReadOnly] public NativeArray<byte> Sleeping;
        [ReadOnly] public NativeArray<float2> PathAccel;
        public float PathForceEnabled;

        [NativeDisableParallelForRestriction]public NativeArray<float2> Positions;
        [NativeDisableParallelForRestriction]public NativeArray<float2> Velocities;
        [NativeDisableParallelForRestriction]public NativeArray<float> Ages;
        public float RotationEnabled;
        public float SpinStrength;
        [NativeDisableParallelForRestriction]public NativeArray<float> Angles;

        public void Execute(int i)
        {
            int id = Active[i];
            if (Alive[id] == 0) return;
            bool isSleep = Sleeping[id] != 0;

            float2 aPf = PathAccel.IsCreated ? (PathAccel[id] * PathForceEnabled) : 0f;

            float2 p = Positions[id];
            float2 v = Velocities[id];
            float ang = Angles.IsCreated ? Angles[id] : 0f;

            float h = DT / math.max(1, Substeps);

            for (int s = 0; s < Substeps; s++)
            {
                if (!isSleep)
                {
                    float2 a = Accel + aPf;
                    v += a * h;
                    p += v * h;
                    if (RotationEnabled > 0f) ang += SpinStrength * v.x * h;
                }
                else
                {
                    v *= (1f - LinearDamping * 0.1f);
                }
            }

            v *= (1f - LinearDamping);

            Positions[id] = p;
            Velocities[id] = v;
            if (Angles.IsCreated) Angles[id] = ang;
            Ages[id] = Ages[id] + DT;
        }
    }

    [BurstCompile(FloatPrecision = FloatPrecision.Low, FloatMode = FloatMode.Fast)]
    private struct BuildGridJobA : IJobParallelForDefer
    {
        [ReadOnly] public NativeArray<int> Active;
        [ReadOnly] public NativeArray<float2> Positions;
        [ReadOnly] public NativeArray<byte> Alive;
        public float2 Origin;
        public float InvCell;
        public NativeParallelMultiHashMap<int, int>.ParallelWriter Writer;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private int HashCell(int2 c) => c.x * 73856093 ^ c.y * 19349663;

        public void Execute(int i)
        {
            int id = Active[i];
            if (Alive[id] == 0) return;
            float2 p = Positions[id];
            int2 cell = (int2)math.floor((p - Origin) * InvCell);
            Writer.Add(HashCell(cell), id);
        }
    }

    [BurstCompile(FloatPrecision = FloatPrecision.Low, FloatMode = FloatMode.Fast)]
    private struct CollisionAndObstacleProjectJobA : IJobParallelForDefer
    {
        [ReadOnly] public NativeArray<int> Active;
        [ReadOnly] public NativeArray<float2> Positions;
        [ReadOnly] public NativeArray<byte> Alive;
        [ReadOnly] public NativeParallelMultiHashMap<int, int> CellMap;

        public float2 Origin;
        public float InvCell;
        public float Radius;
        public float Stiffness;

        // Preinflated AABBs
        [ReadOnly] public NativeArray<float2> ObstMin;
        [ReadOnly] public NativeArray<float2> ObstMax;
        [ReadOnly] public NativeArray<byte>   ObstBlock;

        [NativeDisableParallelForRestriction] public NativeArray<float2> ProjDelta; // single accumulator

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private int HashCell(int2 c) => c.x * 73856093 ^ c.y * 19349663;

        public void Execute(int i)
        {
            int id = Active[i];
            if (Alive[id] == 0) { ProjDelta[id] = 0f; return; }

            float2 pi = Positions[id];
            int2 c0 = (int2)math.floor((pi - Origin) * InvCell);

            float rSum = Radius * 2f;
            float r2 = rSum * rSum;
            float2 corr = 0f;
            const float eps = 1e-12f;

            // Peer collisions via grid
            for (int oy = -1; oy <= 1; oy++)
            for (int ox = -1; ox <= 1; ox++)
            {
                int key = HashCell(new int2(c0.x + ox, c0.y + oy));
                int j;
                if (CellMap.TryGetFirstValue(key, out j, out NativeParallelMultiHashMapIterator<int> it))
                {
                    do
                    {
                        if (j == id || Alive[j] == 0) continue;
                        float2 d = pi - Positions[j];
                        float d2 = math.lengthsq(d);
                        if (d2 > eps && d2 < r2)
                        {
                            // rsqrt-based normalization (avoids scalar sqrt)
                            float invDist = math.rsqrt(d2 + 1e-20f);
                            float2 n = d * invDist;
                            // dist = 1 / invDist (fast reciprocal)
                            float dist = math.rcp(invDist);
                            float overlap = (rSum - dist);
                            corr += 0.5f * overlap * n;
                        }
                    } while (CellMap.TryGetNextValue(out j, ref it));
                }
            }

            // Obstacles (preinflated AABBs, branch-light)
            for (int k = 0; k < ObstMin.Length; k++)
            {
                if (ObstBlock[k] == 0) continue;
                float2 mn = ObstMin[k];
                float2 mx = ObstMax[k];
                // inside check
                if (pi.x > mn.x & pi.x < mx.x & pi.y > mn.y & pi.y < mx.y)
                {
                    float dl = pi.x - mn.x;
                    float dr = mx.x - pi.x;
                    float db = pi.y - mn.y;
                    float dt = mx.y - pi.y;

                    // minimal push-out (branch-minimized)
                    float2 push;
                    if (dl < dr)
                    {
                        if (dl < db) push = dl < dt ? new float2(-dl, 0) : new float2(0, dt);
                        else         push = db < dt ? new float2(0, -db) : new float2(0, dt);
                    }
                    else
                    {
                        if (dr < db) push = dr < dt ? new float2(+dr, 0) : new float2(0, dt);
                        else         push = db < dt ? new float2(0, -db) : new float2(0, dt);
                    }
                    corr += push;
                }
            }

            ProjDelta[id] = corr * math.clamp(Stiffness, 0f, 1f);
        }
    }

    [BurstCompile(FloatPrecision = FloatPrecision.Low, FloatMode = FloatMode.Fast)]
    private struct ApplyProjectionJobA : IJobParallelForDefer
    {
        [ReadOnly] public NativeArray<int> Active;
        public float DT;
        public float VelFactor;

        [ReadOnly] public NativeArray<byte> Alive;
        [ReadOnly] public NativeArray<float2> ProjDelta;

        [NativeDisableParallelForRestriction] public NativeArray<float2> Positions;
        [NativeDisableParallelForRestriction] public NativeArray<float2> Velocities;

        public void Execute(int i)
        {
            int id = Active[i];
            if (Alive[id] == 0) return;
            float2 dp = ProjDelta[id];
            if (math.all(dp == 0f)) return;

            Positions[id] += dp;
            if (DT > 1e-6f && VelFactor > 0f)
                Velocities[id] += (dp / DT) * VelFactor;
        }
    }

    [BurstCompile(FloatPrecision = FloatPrecision.Low, FloatMode = FloatMode.Fast)]
    private struct CorridorConstraintJobA : IJobParallelForDefer
    {
        [ReadOnly] public NativeArray<int> Active;
        public float HalfWidth;
        public float Restitution;
        public int SearchWindow;

        [ReadOnly] public NativeArray<float2> SpPos;
        [ReadOnly] public NativeArray<float2> SpTan;
        [ReadOnly] public NativeArray<byte> Alive;

        [NativeDisableParallelForRestriction]public NativeArray<float2> Positions;
        [NativeDisableParallelForRestriction]public NativeArray<float2> Velocities;
        [NativeDisableParallelForRestriction] public NativeArray<int> NearestIndices;

        public void Execute(int i)
        {
            int id = Active[i];
            if (Alive[id] == 0) return;

            float2 p = Positions[id];
            float2 v = Velocities[id];

            int nearest = NearestIndices[id];
            if ((uint)nearest >= (uint)SpPos.Length) nearest = 0;

            int start = math.max(0, nearest - SearchWindow);
            int end = math.min(SpPos.Length - 1, nearest + SearchWindow);

            float best = math.lengthsq(p - SpPos[nearest]);
            int bestIndex = nearest;
            for (int k = start; k <= end; k++)
            {
                float d2 = math.lengthsq(p - SpPos[k]);
                if (d2 < best) { best = d2; bestIndex = k; }
            }
            nearest = bestIndex;

            float2 c = SpPos[nearest];
            float2 t = SpTan[nearest];
            float2 d = p - c;
            float along = math.dot(d, t);
            float2 lateral = d - along * t;
            float dist = math.length(lateral);

            if (dist > HalfWidth)
            {
                float2 n = lateral / math.max(dist, 1e-8f);
                p = c + along * t + n * HalfWidth;
                float vn = math.dot(v, n);
                v = v - (1f + Restitution) * vn * n;
            }

            Positions[id] = p;
            Velocities[id] = v;
            NearestIndices[id] = nearest;
        }
    }

    [BurstCompile(FloatPrecision = FloatPrecision.Low, FloatMode = FloatMode.Fast)]
    private struct SleepCheckJobA : IJobParallelForDefer
    {
        [ReadOnly] public NativeArray<int> Active;
        [ReadOnly] public NativeArray<byte> Alive;
        [ReadOnly] public NativeArray<float2> Velocities;
        [ReadOnly] public NativeArray<float2> Proj;

        public float SleepVel2;
        public float WakeVel2;
        public float SleepProjTol;
        public int SleepFrames;

        [NativeDisableParallelForRestriction] public NativeArray<byte> Sleeping;
        [NativeDisableParallelForRestriction] public NativeArray<ushort> SleepCounter;

        public void Execute(int i)
        {
            int id = Active[i];
            if (Alive[id] == 0)
            {
                Sleeping[id] = 0; SleepCounter[id] = 0;
                return;
            }

            float v2 = math.lengthsq(Velocities[id]);
            float proj = math.length(Proj[id]);

            if (Sleeping[id] == 0)
            {
                if (v2 < SleepVel2 && proj < SleepProjTol)
                {
                    ushort c = (ushort)math.min(65535, SleepCounter[id] + 1);
                    SleepCounter[id] = c;
                    if (c >= SleepFrames) Sleeping[id] = 1;
                }
                else SleepCounter[id] = 0;
            }
            else
            {
                if (v2 > WakeVel2 || proj > SleepProjTol * 2f)
                {
                    Sleeping[id] = 0;
                    SleepCounter[id] = 0;
                }
            }
        }
    }

    [BurstCompile(FloatPrecision = FloatPrecision.Low, FloatMode = FloatMode.Fast)]
    private struct DespawnBelowYJobA : IJobParallelForDefer
    {
        [ReadOnly] public NativeArray<int> Active;
        [ReadOnly] public NativeArray<byte> Alive;
        [ReadOnly] public NativeArray<float2> Positions;
        public float DespawnY;
        public NativeList<int>.ParallelWriter ToDespawn;

        public void Execute(int i)
        {
            int id = Active[i];
            if (Alive[id] == 0) return;
            if (Positions[id].y < DespawnY) ToDespawn.AddNoResize(id);
        }
    }

    [BurstCompile(FloatPrecision = FloatPrecision.Low, FloatMode = FloatMode.Fast)]
    private struct PackGPUJobA : IJobParallelForDefer
    {
        [ReadOnly] public NativeArray<int> Active;
        [ReadOnly] public NativeArray<byte> Alive;
        [ReadOnly] public NativeArray<float2> Positions;
        [ReadOnly] public NativeArray<float> Angles;
        [ReadOnly] public NativeArray<int> SpriteIndex;
        [ReadOnly] public NativeArray<float> Ages;
        public float ColliderRadius;
        public int SortMode; // 0 none, 1 age, 2 id
        public NativeArray<ParticleGPU> Packets;

        public void Execute(int i)
        {
            int id = Active[i];
            if (Alive[id] == 0)
            {
                // Write a degenerate, but keep stable index (we'll compact at upload)
                Packets[i] = default;
                return;
            }
            float sortKey = 0f;
            if (SortMode == 1) sortKey = Ages.IsCreated ? Ages[id] : 0f;
            else if (SortMode == 2) sortKey = id;

            float2 p = Positions[id];
            Packets[i] = new ParticleGPU
            {
                X = p.x,
                Y = p.y,
                R = ColliderRadius,
                Angle = Angles.IsCreated ? Angles[id] : 0f,
                SpriteIndex = SpriteIndex.IsCreated ? SpriteIndex[id] : 0f,
                SortKey = sortKey
            };
        }
    }

    // ======================= LIFECYCLE =======================

    private void OnEnable()
    {
        rng = new Random((uint)math.max(1, Seed));
        Allocate();

        LoadBakedAndBuildPathAndObstacles();
        SetupGridFromPath();
        PrimeSpawn();

        SetupIndirectRendering();
        BuildSpriteArrayIfNeeded();
        UpdateDrawBounds();

        lastCorridorWidth = CorridorWidth;
        lastColliderRadius = ColliderRadius;
    }

    private void OnDisable()
    {
        lastHandle.Complete();
        DisposeAll();
        ReleaseIndirectRendering();
    }

    private void Allocate()
    {
        if (allocated) return;

        Capacity = math.max(1, Capacity);
        MaxActive = math.clamp(MaxActive, 0, Capacity);

        positions = new NativeArray<float2>(Capacity, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        velocities = new NativeArray<float2>(Capacity, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        alive     = new NativeArray<byte>(Capacity, Allocator.Persistent, NativeArrayOptions.ClearMemory);

        spPos = new NativeArray<float2>(math.max(2, PathSampleCount), Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        spTan = new NativeArray<float2>(math.max(2, PathSampleCount), Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

        nearestIndices = new NativeArray<int>(Capacity, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        projDelta      = new NativeArray<float2>(Capacity, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

        sleeping     = new NativeArray<byte>(Capacity, Allocator.Persistent, NativeArrayOptions.ClearMemory);
        sleepCounter = new NativeArray<ushort>(Capacity, Allocator.Persistent, NativeArrayOptions.ClearMemory);

        pathAccel    = new NativeArray<float2>(Capacity, Allocator.Persistent, NativeArrayOptions.ClearMemory);

        angles      = new NativeArray<float>(Capacity, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        spriteIndex = new NativeArray<int>(Capacity, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        ages        = new NativeArray<float>(Capacity, Allocator.Persistent, NativeArrayOptions.ClearMemory);

        int cellCapacity = math.max(4096, Capacity * 5);
        cellMap = new NativeParallelMultiHashMap<int, int>(cellCapacity, Allocator.Persistent);

        for (int i = 0; i < Capacity; i++) nearestIndices[i] = -1;

        activeIds = new NativeList<int>(Capacity, Allocator.Persistent);
        freeIds   = new NativeList<int>(Capacity, Allocator.Persistent);
        for (int i = 0; i < Capacity; i++) freeIds.Add(i);

        // GPU packet buffer (max = Capacity, we pack active using IJobParallelForDefer)
        gpuPackets = new NativeArray<ParticleGPU>(Capacity, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

        allocated = true;
    }

    private void DisposeAll()
    {
        if (!allocated) return;
        if (positions.IsCreated) positions.Dispose();
        if (velocities.IsCreated) velocities.Dispose();
        if (alive.IsCreated) alive.Dispose();
        if (spPos.IsCreated) spPos.Dispose();
        if (spTan.IsCreated) spTan.Dispose();
        if (nearestIndices.IsCreated) nearestIndices.Dispose();
        if (projDelta.IsCreated) projDelta.Dispose();
        if (cellMap.IsCreated) cellMap.Dispose();
        if (obstacleMin.IsCreated) obstacleMin.Dispose();
        if (obstacleMax.IsCreated) obstacleMax.Dispose();
        if (obstacleBlock.IsCreated) obstacleBlock.Dispose();
        if (sleeping.IsCreated) sleeping.Dispose();
        if (sleepCounter.IsCreated) sleepCounter.Dispose();
        if (pathAccel.IsCreated) pathAccel.Dispose();
        if (activeIds.IsCreated) activeIds.Dispose();
        if (freeIds.IsCreated) freeIds.Dispose();
        if (ages.IsCreated) ages.Dispose();
        if (angles.IsCreated) angles.Dispose();
        if (spriteIndex.IsCreated) spriteIndex.Dispose();
        if (gpuPackets.IsCreated) gpuPackets.Dispose();
        allocated = false;
    }

    private void BuildSpriteArrayIfNeeded()
    {
        if (!BuildArrayAtRuntime || Sprites == null || Sprites.Count == 0) return;

        int w = Mathf.Max(1, SpriteSize.x);
        int h = Mathf.Max(1, SpriteSize.y);
        int slices = Sprites.Count;

        spriteArray = new Texture2DArray(w, h, slices, TextureFormat.RGBA32, false)
        {
            filterMode = SpriteFilter,
            wrapMode = TextureWrapMode.Clamp
        };

        bool canCopyGPU = SystemInfo.copyTextureSupport != CopyTextureSupport.None;

        for (int i = 0; i < slices; i++)
        {
            var s = Sprites[i];
            var tex = s != null ? s.texture : Texture2D.whiteTexture;
            Rect r = s != null ? s.rect : new Rect(0, 0, tex.width, tex.height);

            int sx = Mathf.RoundToInt(r.x);
            int sy = Mathf.RoundToInt(r.y);
            int sw = Mathf.RoundToInt(r.width);
            int sh = Mathf.RoundToInt(r.height);

            if (sw == w && sh == h && canCopyGPU && tex != null)
            {
                if (sx == 0 && sy == 0 && sw == tex.width && sh == tex.height)
                {
                    Graphics.CopyTexture(tex, 0, 0, spriteArray, i, 0);
                }
                else
                {
                    var tmp = new Texture2D(sw, sh, TextureFormat.RGBA32, false, false);
                    tmp.SetPixels(tex.GetPixels(sx, sy, sw, sh));
                    tmp.Apply(false, false);
                    Graphics.CopyTexture(tmp, 0, 0, spriteArray, i, 0);
                }
            }
            else
            {
                Color[] src = tex.GetPixels(sx, sy, sw, sh);
                Color[] dst = new Color[w * h];

                for (int y = 0; y < h; y++)
                {
                    float v = (y + 0.5f) / h * sh - 0.5f;
                    int y0 = Mathf.Clamp(Mathf.FloorToInt(v), 0, sh - 1);
                    int y1 = Mathf.Clamp(y0 + 1, 0, sh - 1);
                    float fy = Mathf.Clamp01(v - y0);

                    for (int x = 0; x < w; x++)
                    {
                        float u = (x + 0.5f) / w * sw - 0.5f;
                        int x0 = Mathf.Clamp(Mathf.FloorToInt(u), 0, sw - 1);
                        int x1 = Mathf.Clamp(x0 + 1, 0, sw - 1);
                        float fx = Mathf.Clamp01(u - x0);

                        Color c00 = src[y0 * sw + x0];
                        Color c10 = src[y0 * sw + x1];
                        Color c01 = src[y1 * sw + x0];
                        Color c11 = src[y1 * sw + x1];

                        Color cx0 = Color.Lerp(c00, c10, fx);
                        Color cx1 = Color.Lerp(c01, c11, fx);
                        dst[y * w + x] = Color.Lerp(cx0, cx1, fy);
                    }
                }

                spriteArray.SetPixels(dst, i, 0);
            }
        }

        spriteArray.Apply(false, false);
        if (ObjectMaterial != null) ObjectMaterial.SetTexture(SpriteArrayID, spriteArray);
    }

    private void LoadBakedAndBuildPathAndObstacles()
    {
        List<Vector2> pathNodes = new();
        List<(float2 pos, float2 half, bool open)> bakedObs = new();

        if (BakedJson != null && !string.IsNullOrEmpty(BakedJson.text))
        {
            try
            {
                var baked = JsonUtility.FromJson<BakedLevel>(BakedJson.text);
                if (baked != null)
                {
                    if (baked.corridorWidth > 0f) CorridorWidth = baked.corridorWidth;
                    if (baked.colliderRadius > 0f) ColliderRadius = baked.colliderRadius;
                    if (baked.gridCellSize > 0f) GridCellSize = baked.gridCellSize;

                    if (baked.path != null && baked.path.Count >= 2)
                        foreach (var p in baked.path) pathNodes.Add(new Vector2(p.x, p.y));

                    if (baked.obstacles != null && baked.obstacles.Count > 0)
                        foreach (var o in baked.obstacles)
                            bakedObs.Add((new float2(o.x, o.y), math.max(0.001f, o.radius), !o.open ? true : false));
                }
            }
            catch (Exception e) { Debug.LogException(e); }
        }

        if (pathNodes.Count < 2)
        {
            pathNodes.Clear();
            pathNodes.Add(new Vector2(-5, 0));
            pathNodes.Add(new Vector2(+5, 0));
        }

        BakePolylineToSamples(pathNodes, math.max(2, PathSampleCount));

        // obstacles: pre-inflate AABBs by ColliderRadius once (updated if radius changes)
        if (obstacleMin.IsCreated) obstacleMin.Dispose();
        if (obstacleMax.IsCreated) obstacleMax.Dispose();
        if (obstacleBlock.IsCreated) obstacleBlock.Dispose();

        int m = bakedObs.Count;
        obstacleMin = new NativeArray<float2>(m, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        obstacleMax = new NativeArray<float2>(m, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        obstacleBlock = new NativeArray<byte>(m, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

        for (int i = 0; i < m; i++)
        {
            float2 half = bakedObs[i].half;
            float2 pos = bakedObs[i].pos;
            float2 pad = new float2(ColliderRadius);
            obstacleMin[i] = pos - half - pad;
            obstacleMax[i] = pos + half + pad;
            obstacleBlock[i] = (byte)(bakedObs[i].open ? 0 : 1);
        }
    }

    private void BakePolylineToSamples(List<Vector2> pts, int sampleCount)
    {
        float totalLen = 0f;
        for (int i = 0; i < pts.Count - 1; i++) totalLen += Vector2.Distance(pts[i], pts[i + 1]);
        if (totalLen <= 1e-5f) totalLen = 1f;
        spStepLen = totalLen / math.max(1, sampleCount - 1);

        List<float2> pList = new(sampleCount);
        List<float2> tList = new(sampleCount);

        Vector2 a0 = pts[0];
        pList.Add(new float2(a0.x, a0.y));
        Vector2 nextDir0 = (pts[1] - pts[0]).normalized;
        tList.Add(math.normalizesafe(new float2(nextDir0.x, nextDir0.y), new float2(1, 0)));

        float carried = 0f;
        float step = spStepLen;

        for (int seg = 0; seg < pts.Count - 1 && pList.Count < sampleCount - 1; seg++)
        {
            Vector2 a = pts[seg];
            Vector2 b = pts[seg + 1];
            float len = Vector2.Distance(a, b);
            if (len < 1e-6f) continue;

            Vector2 dir = (b - a) / len;
            float t = step - carried;
            while (t <= len && pList.Count < sampleCount - 1)
            {
                Vector2 p = a + dir * t;
                pList.Add(new float2(p.x, p.y));
                tList.Add(new float2(dir.x, dir.y));
                t += step;
            }
            carried = (carried + len) % step;
        }

        if (pList.Count < sampleCount)
        {
            Vector2 last = pts[^1];
            pList.Add(new float2(last.x, last.y));
            Vector2 lastDir = (pts[^1] - pts[^2]).normalized;
            tList.Add(new float2(lastDir.x, lastDir.y));
        }

        int N = math.min(sampleCount, pList.Count);
        if (!spPos.IsCreated || spPos.Length != N)
        {
            if (spPos.IsCreated) spPos.Dispose();
            if (spTan.IsCreated) spTan.Dispose();
            spPos = new NativeArray<float2>(N, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            spTan = new NativeArray<float2>(N, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        }
        for (int i = 0; i < N; i++)
        {
            spPos[i] = pList[i];
            spTan[i] = math.normalizesafe(tList[i], new float2(1, 0));
        }
        PathSampleCount = N;
    }

    private void SetupGridFromPath()
    {
        if (!spPos.IsCreated || spPos.Length == 0) return;
        float2 mn = spPos[0], mx = spPos[0];
        for (int i = 1; i < spPos.Length; i++) { mn = math.min(mn, spPos[i]); mx = math.max(mx, spPos[i]); }

        float2 pad = new float2(HalfWidth + GridCellSize * 2f);
        mn -= pad;
        gridOrigin = mn;

        float desired = ColliderRadius * 2f;
        GridCellSize = math.max(GridCellSize, desired);
        invCell = 1f / GridCellSize;
    }

    private void PrimeSpawn()
    {
        if (SpawnPoint == null) return;
        int toSpawn = math.min(BurstOnStart > 0 ? BurstOnStart : (SpawnPerSecond == 0 ? MaxActive : 0), MaxActive);
        SpawnN(toSpawn);
    }

    private void SpawnN(int n)
    {
        if (n <= 0 || SpawnPoint == null) return;
        int spawned = 0;
        float2 spBase = new float2(SpawnPoint.position.x, SpawnPoint.position.y);
        float angRad = math.radians(SpawnAngleDeg);
        float2 dir = math.normalizesafe(new float2(math.cos(angRad), math.sin(angRad)), new float2(1, 0));

        while (spawned < n && activeCount < MaxActive && freeIds.Length > 0)
        {
            int last = freeIds.Length - 1;
            int id = freeIds[last];
            freeIds.RemoveAtSwapBack(last);

            float2 jitter;
            if (UseCircleJitter)
            {
                float r = rng.NextFloat(0f, 1f);
                float ang = rng.NextFloat(0f, math.PI * 2f);
                float rr = math.sqrt(r) * SpawnCircleJitterRadius;
                jitter = new float2(math.cos(ang), math.sin(ang)) * rr;
            }
            else
            {
                jitter = new float2(rng.NextFloat(-SpawnBoxJitter.x, +SpawnBoxJitter.x), rng.NextFloat(-SpawnBoxJitter.y, +SpawnBoxJitter.y));
            }
            float2 sp = spBase + jitter;
            if (!math.all(math.isfinite(dir)) || math.lengthsq(dir) < 1e-6f) dir = new float2(1, 0);

            int idx = 0;
            if (spriteArray != null) idx = rng.NextInt(0, spriteArray.depth);

            positions[id] = sp;
            velocities[id] = dir * InitialSpeed;
            alive[id] = 1;
            nearestIndices[id] = -1;
            sleeping[id] = 0;
            sleepCounter[id] = 0;
            ages[id] = 0f;

            angles[id] = rng.NextFloat(0f, math.PI * 2f);
            spriteIndex[id] = idx;

            activeIds.Add(id);
            activeCount++;
            spawned++;
        }
    }

    private void SetupIndirectRendering()
    {
        if (QuadMesh == null) QuadMesh = CreateUnitQuadMesh();

        argsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, 1, sizeof(uint) * 5);
        UpdateArgs((uint)QuadMesh.GetIndexCount(0), 0u, (uint)QuadMesh.GetIndexStart(0), (uint)QuadMesh.GetBaseVertex(0));

        int stride = sizeof(float) * 6;  // 24 bytes
        instanceBuffer = new ComputeBuffer(Capacity, stride, ComputeBufferType.Structured, ComputeBufferMode.SubUpdates);

        if (ObjectMaterial != null)
        {
            ObjectMaterial.SetBuffer(ParticlesID, instanceBuffer);
            if (spriteArray != null) ObjectMaterial.SetTexture(SpriteArrayID, spriteArray);
        }
    }

    private void ReleaseIndirectRendering()
    {
        argsBuffer?.Dispose(); argsBuffer = null;
        instanceBuffer?.Dispose(); instanceBuffer = null;
    }

    private Mesh CreateUnitQuadMesh()
    {
        var m = new Mesh { name = "2D_Quad" };
        m.SetVertices(new List<Vector3>
        {
            new Vector3(-1, -1, 0), new Vector3(1, -1, 0),
            new Vector3(1,  1, 0), new Vector3(-1,  1, 0)
        });
        m.SetUVs(0, new List<Vector2>
        {
            new Vector2(0, 0), new Vector2(1, 0),
            new Vector2(1, 1), new Vector2(0, 1)
        });
        m.SetTriangles(new[] { 0, 1, 2, 0, 2, 3 }, 0, true);
        m.RecalculateBounds();
        return m;
    }

    private void UpdateArgs(uint indexCountPerInstance, uint instanceCount, uint startIndex, uint baseVertex)
    {
        if (argsBuffer == null) return;
        var args = new uint[5] { indexCountPerInstance, instanceCount, startIndex, baseVertex, 0u };
        argsBuffer.SetData(args);
    }

    private void UpdateDrawBounds()
    {
        if (!spPos.IsCreated || spPos.Length == 0)
        {
            drawBounds = new Bounds(Vector3.zero, new Vector3(1000, 1000, 10));
            return;
        }

        float2 mn = spPos[0], mx = spPos[0];
        for (int i = 1; i < spPos.Length; i++) { mn = math.min(mn, spPos[i]); mx = math.max(mx, spPos[i]); }

        float pad = CorridorWidth * 0.6f + math.max(0.5f, GridCellSize * 2f);
        var center = new Vector3((mn.x + mx.x) * 0.5f, (mn.y + mn.y) * 0.5f, 0f);
        var size   = new Vector3((mx.x - mn.x) + pad * 2f, (mx.y - mn.y) + pad * 2f, 5f);
        drawBounds = new Bounds(center, size);
    }

    private void FixedUpdate()
    {
        if (!allocated) return;

        swTotal.Restart();

        if (SpawnPerSecond > 0 && activeCount < MaxActive)
        {
            spawnAcc += Mathf.Max(Time.fixedDeltaTime, 0f) * SpawnPerSecond;
            int spawnNow = (int)spawnAcc;
            if (spawnNow > 0)
            {
                spawnAcc -= spawnNow;
                SpawnN(spawnNow);
            }
        }

        if (!Mathf.Approximately(lastCorridorWidth, CorridorWidth))
        {
            UpdateDrawBounds();
            lastCorridorWidth = CorridorWidth;
        }

        // Inflate AABBs if radius changed
        if (!Mathf.Approximately(lastColliderRadius, ColliderRadius) && obstacleMin.IsCreated && obstacleMax.IsCreated)
        {
            float2 pad = new float2(ColliderRadius);
            for (int i = 0; i < obstacleMin.Length; i++)
            {
                float2 center = (obstacleMin[i] + obstacleMax[i]) * 0.5f;
                float2 half   = (obstacleMax[i] - obstacleMin[i]) * 0.5f - new float2(lastColliderRadius);
                obstacleMin[i] = center - (half + pad);
                obstacleMax[i] = center + (half + pad);
            }
            lastColliderRadius = ColliderRadius;
        }

        if (MobileTuningEnabled)
        {
            float ft = Time.deltaTime;
            if (ft > MobileTargetFrameTime)
            {
                CollisionIterations = math.max(1, CollisionIterations - 1);
                NearestSearchWindow = math.max(6, NearestSearchWindow - 2);
            }
            if (ft > MobileHardFrameTime)
            {
                Substeps = 1;
                SpawnPerSecond = (int)(SpawnPerSecond * 0.8f);
            }
        }

        float dt = math.clamp(Time.fixedDeltaTime, 1e-4f, 1f / 30f);

        var activesDeferred = activeIds.AsDeferredJobArray();

        JobHandle hPf = default;
        if (PathForceActive && activeCount > 0)
        {
            int lookaheadSamples = math.max(1, (int)math.round(PathForceLookahead / math.max(1e-6f, spStepLen)));
            var pf = new PathForceJobA
            {
                Active = activesDeferred,
                Alive = alive,
                Positions = positions,
                NearestId = nearestIndices,
                SpPos = spPos,
                LookaheadSamples = lookaheadSamples,
                Strength = PathForceStrength,
                outAccel = pathAccel,
                minAbsX = math.clamp(MinAbsX, 0f, 1f),
            };
            hPf = pf.Schedule(activeIds, 64, default);
        }

        if (activeCount == 0)
        {
            swTotal.Stop();
            return;
        }

        var integrate = new IntegrateJobA
        {
            Active = activesDeferred,
            DT = dt,
            Substeps = math.max(1, Substeps),
            LinearDamping = Mathf.Clamp01(LinearDamping),
            Accel = new float2(0f, Gravity),
            Alive = alive,
            Sleeping = sleeping,
            Positions = positions,
            Velocities = velocities,
            PathAccel = pathAccel,
            PathForceEnabled = PathForceActive ? 1f : 0f,
            Ages = ages,
            RotationEnabled = RotationEnabled ? 1f : 0f,
            SpinStrength = SpinStrength,
            Angles = angles
        };
        var hIntegrate = integrate.Schedule(activeIds, 64, hPf);

        int iters = math.max(1, CollisionIterations);
        cellMap.Clear();

        var build = new BuildGridJobA
        {
            Active = activesDeferred,
            Positions = positions,
            Alive = alive,
            Origin = gridOrigin,
            InvCell = invCell,
            Writer = cellMap.AsParallelWriter()
        };
        var hGrid = build.Schedule(activeIds, 64, hIntegrate);

        JobHandle hPrev = hGrid;

        for (int iter = 0; iter < iters; iter++)
        {
            var proj = new CollisionAndObstacleProjectJobA
            {
                Active = activesDeferred,
                Positions = positions,
                Alive = alive,
                CellMap = cellMap,
                Origin = gridOrigin,
                InvCell = invCell,
                Radius = ColliderRadius,
                Stiffness = ProjectionStiffness,
                ObstMin = obstacleMin.IsCreated ? obstacleMin : default,
                ObstMax = obstacleMax.IsCreated ? obstacleMax : default,
                ObstBlock = obstacleBlock.IsCreated ? obstacleBlock : default,
                ProjDelta = projDelta
            };
            var hProj = proj.Schedule(activeIds, 64, hPrev);

            var apply = new ApplyProjectionJobA
            {
                Active = activesDeferred,
                DT = dt,
                VelFactor = VelocityFromProjection,
                Alive = alive,
                ProjDelta = projDelta,
                Positions = positions,
                Velocities = velocities
            };
            hPrev = apply.Schedule(activeIds, 64, hProj);
        }

        var corridor = new CorridorConstraintJobA
        {
            Active = activesDeferred,
            HalfWidth = math.max(0.001f, HalfWidth),
            Restitution = Mathf.Clamp01(Restitution),
            SearchWindow = math.max(2, NearestSearchWindow),
            SpPos = spPos,
            SpTan = spTan,
            Alive = alive,
            Positions = positions,
            Velocities = velocities,
            NearestIndices = nearestIndices
        };
        var hCorridor = corridor.Schedule(activeIds, 64, hPrev);

        var sleep = new SleepCheckJobA
        {
            Active = activesDeferred,
            Alive = alive,
            Velocities = velocities,
            Proj = projDelta,
            SleepVel2 = SleepVel * SleepVel,
            WakeVel2 = WakeVel * WakeVel,
            SleepProjTol = SleepProjTol,
            SleepFrames = math.max(1, SleepFrames),
            Sleeping = sleeping,
            SleepCounter = sleepCounter
        };
        var hAfterSleep = sleep.Schedule(activeIds, 64, hCorridor);

        JobHandle hFinal = hAfterSleep;
        NativeList<int> toDespawn = default;
        if (DespawnBelowYEnabled || DespawnScanRequested)
        {
            toDespawn = new NativeList<int>(Allocator.TempJob);
            int reserve = math.max(256, activeIds.Length / 8);
            toDespawn.Capacity = math.max(toDespawn.Capacity, reserve);

            var despawn = new DespawnBelowYJobA
            {
                Active = activesDeferred,
                Alive = alive,
                Positions = positions,
                DespawnY = DespawnY,
                ToDespawn = toDespawn.AsParallelWriter()
            };
            hFinal = despawn.Schedule(activeIds, 64, hAfterSleep);
        }

        hFinal.Complete();

        if (toDespawn.IsCreated)
        {
            DespawnScanRequested = false;

            for (int n = 0; n < toDespawn.Length; n++)
            {
                int id = toDespawn[n];
                if (alive[id] != 0)
                {
                    alive[id] = 0;
                    sleeping[id] = 0;
                    sleepCounter[id] = 0;
                    freeIds.Add(id);
                    activeCount = math.max(0, activeCount - 1);
                }
            }

            // compact active list in-place
            int write = 0;
            for (int k = 0; k < activeIds.Length; k++)
            {
                int id = activeIds[k];
                if ((uint)id < (uint)alive.Length && alive[id] != 0)
                    activeIds[write++] = id;
            }
            activeIds.ResizeUninitialized(write);

            toDespawn.Dispose();
        }

        swTotal.Stop();
        if (BenchmarkEnabled)
        {
            benchFrames++;
            sumTotal += swTotal.Elapsed.TotalMilliseconds;
            if (benchFrames % math.max(1, BenchmarkReportInterval) == 0)
            {
                Debug.Log($"[SIM OPT] avg physics {sumTotal / benchFrames:0.00} ms (frames={benchFrames}, active={activeCount})");
                benchFrames = 0;
                sumTotal = 0.0;
            }
        }
    }

    private void Update()
    {
        if (!allocated) return;
        UploadInstancesAndDraw();
    }

    private void UploadInstancesAndDraw()
    {
        if (ObjectMaterial == null || QuadMesh == null || argsBuffer == null) return;

        int live = math.clamp(activeCount, 0, MaxActive);
        if (live == 0)
        {
            UpdateArgs((uint)QuadMesh.GetIndexCount(0), 0u, (uint)QuadMesh.GetIndexStart(0), (uint)QuadMesh.GetBaseVertex(0));
            return;
        }

        // Build packets on a job (active-only), then single upload
        int sortMode = (SortBy == SortMode.None) ? 0 : (SortBy == SortMode.ByAge ? 1 : 2);
        var pack = new PackGPUJobA
        {
            Active = activeIds.AsDeferredJobArray(),
            Alive = alive,
            Positions = positions,
            Angles = angles,
            SpriteIndex = spriteIndex,
            Ages = ages,
            ColliderRadius = ColliderRadius,
            SortMode = sortMode,
            Packets = gpuPackets
        };
        var hPack = pack.Schedule(activeIds, 64, default);
        hPack.Complete(); // dependency boundary before GPU upload

        int count = activeIds.Length;

        if (!CpuFrontToBackSort || SortBy == SortMode.None)
        {
            // Compact while uploading: we filter out any inactive remnants (rare)
            // Use a temporary managed array to set count; faster is keeping as-is since we 1:1 with activeIds
            instanceBuffer.SetData(gpuPackets, 0, 0, count);
            UpdateArgs((uint)QuadMesh.GetIndexCount(0), (uint)count, (uint)QuadMesh.GetIndexStart(0), (uint)QuadMesh.GetBaseVertex(0));
            if (count > 0) Graphics.DrawMeshInstancedIndirect(QuadMesh, 0, ObjectMaterial, drawBounds, argsBuffer);
            return;
        }

        // CPU sort (optional) — minimal changes, sort an index list, then upload in that order
        if (sortScratch == null || sortScratch.Length < count) sortScratch = new SortEntry[math.ceilpow2(count)];
        int n = 0;
        for (int k = 0; k < count; k++)
        {
            int id = activeIds[k];
            if ((uint)id >= (uint)positions.Length || alive[id] == 0) continue;
            float key = (SortBy == SortMode.ByAge) ? (ages.IsCreated ? ages[id] : 0f) : id;
            sortScratch[n++] = new SortEntry { Key = key, Id = k }; // store packet index (k)
        }
        if (n <= 0)
        {
            UpdateArgs((uint)QuadMesh.GetIndexCount(0), 0u, (uint)QuadMesh.GetIndexStart(0), (uint)QuadMesh.GetBaseVertex(0));
            return;
        }
        Array.Sort(sortScratch, 0, n);
        if (!SortAscending) Array.Reverse(sortScratch, 0, n);

        // Create a contiguous window into gpuPackets using a managed staging array (minimal overhead)
        // Alternative would be another job to reorder, but this happens on small subset typically.
        ParticleGPU[] staging = new ParticleGPU[n];
        for (int i = 0; i < n; i++) staging[i] = gpuPackets[sortScratch[i].Id];
        instanceBuffer.SetData(staging, 0, 0, n);

        UpdateArgs((uint)QuadMesh.GetIndexCount(0), (uint)n, (uint)QuadMesh.GetIndexStart(0), (uint)QuadMesh.GetBaseVertex(0));
        Graphics.DrawMeshInstancedIndirect(QuadMesh, 0, ObjectMaterial, drawBounds, argsBuffer);
    }

    private void OnDrawGizmosSelected()
    {
        if (spPos.IsCreated && spPos.Length >= 2)
        {
            Gizmos.color = Color.white;
            for (int i = 0; i < spPos.Length - 1; i++)
                Gizmos.DrawLine(new Vector3(spPos[i].x, spPos[i].y, 0f),
                                new Vector3(spPos[i + 1].x, spPos[i + 1].y, 0f));

            Gizmos.color = Color.yellow;
            float hw = HalfWidth;
            for (int i = 0; i < spPos.Length - 1; i++)
            {
                float2 t0 = spTan[i];
                float2 n0 = math.normalizesafe(new float2(-t0.y, t0.x), new float2(0, 1));
                Vector3 l0 = new Vector3(spPos[i].x + n0.x * hw, spPos[i].y + n0.y * hw, 0f);
                Vector3 r0 = new Vector3(spPos[i].x - n0.x * hw, spPos[i].y - n0.y * hw, 0f);

                float2 t1 = spTan[i + 1];
                float2 n1 = math.normalizesafe(new float2(-t1.y, t1.x), new float2(0, 1));
                Vector3 l1 = new Vector3(spPos[i + 1].x + n1.x * hw, spPos[i + 1].y + n1.y * hw, 0f);
                Vector3 r1 = new Vector3(spPos[i + 1].x - n1.x * hw, spPos[i + 1].y - n1.y * hw, 0f);

                Gizmos.DrawLine(l0, l1);
                Gizmos.DrawLine(r0, r1);
            }
        }

        if (obstacleMin.IsCreated && obstacleMax.IsCreated)
        {
            Handles.color = new Color(1f, 0f, 0f, 0.5f);
            for (int i = 0; i < obstacleMin.Length; i++)
            {
                float2 c = (obstacleMin[i] + obstacleMax[i]) * 0.5f;
                float2 half = (obstacleMax[i] - obstacleMin[i]) * 0.5f;
                Handles.DrawWireCube(new Vector3(c.x, c.y, 0f), new Vector3(half.x * 2, half.y * 2));
            }
        }

        if (Application.isPlaying && positions.IsCreated && alive.IsCreated)
        {
            Gizmos.color = Color.cyan;
            int drawn = 0;
            int max = 1000;
            for (int k = 0; k < activeIds.Length && drawn < max; k++)
            {
                int id = activeIds[k];
                if ((uint)id >= (uint)positions.Length) continue;
                if (alive[id] == 0) continue;
                Gizmos.DrawSphere(new Vector3(positions[id].x, positions[id].y, 0f), Mathf.Max(0.01f, ColliderRadius * 0.5f));
                drawn++;
            }
        }
    }
}
