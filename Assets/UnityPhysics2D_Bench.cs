using System;
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.Splines;
using Debug = UnityEngine.Debug;

[DefaultExecutionOrder(-50)]
public class UnityPhysics2D_Bench : MonoBehaviour
{
    [Header("Spline Corridor")]
    public SplineContainer SplineContainer;
    public float CorridorWidth = 6f;
    public float WallThickness = 0.1f;
    public int SplineSampleCount = 256;

    [Header("Spawner")]
    public int Capacity = 10000;
    public int SpawnPerSecond = 1000;
    public int BurstOnStart = 0;
    public int MaxActive = 10000;
    public float ColliderRadius = 0.06f;
    public Transform SpawnPoint;
    public Vector2 vecA = new Vector2(-1, 1);
    public Vector2 vecB = new Vector2(-1, 1);

    [Header("Physics2D")]
    public Vector2 Gravity = new Vector2(0, -9.81f);
    public float LinearDamping = 0.01f;
    public float Restitution = 0.9f;

    [Header("Sprite")]
    public int SpritePixels = 32;

    [Header("Benchmark")]
    public bool BenchmarkEnabled = true;
    public int BenchmarkReportInterval = 120;

    private Texture2D _circleTex;
    private Sprite _circleSprite;
    private PhysicsMaterial2D _mat;
    private List<Rigidbody2D> _bodies;
    private List<GameObject> _pool;
    private int _active;
    private float _spawnAcc;
    private System.Random _rng;

    // Scripted simulation
    private double _accum;
    private readonly Stopwatch _swSim = new Stopwatch();
    private readonly Stopwatch _swTotal = new Stopwatch();
    private double _sumTotal, _sumSim, _sumUpload;
    private int _frames;

    private void OnEnable()
    {
        Physics2D.simulationMode = SimulationMode2D.Script;
        Physics2D.gravity = Gravity;
        _rng = new System.Random(12345);
        _mat = new PhysicsMaterial2D("BenchMat"){ bounciness = Mathf.Clamp01(Restitution), friction = 0f };
        _bodies = new List<Rigidbody2D>(Capacity);
        _pool = new List<GameObject>(Capacity);

        BuildCorridor();
        BuildSprite();
        PrewarmPool();
        PrimeSpawn();
    }

    private void OnDisable()
    {
        Physics2D.simulationMode = SimulationMode2D.FixedUpdate;
        foreach (var go in _pool) if (go) Destroy(go);
        _pool.Clear();
        _bodies.Clear();
    }

    private void BuildSprite()
    {
        int s = Mathf.Max(8, SpritePixels);
        _circleTex = new Texture2D(s, s, TextureFormat.RGBA32, false);
        var px = new Color32[s * s];
        float r = (s - 2) * 0.5f;
        float cx = (s - 1) * 0.5f, cy = (s - 1) * 0.5f;
        for (int y = 0; y < s; y++)
        for (int x = 0; x < s; x++)
        {
            float dx = x - cx, dy = y - cy;
            float d2 = dx * dx + dy * dy;
            px[y * s + x] = d2 <= r * r ? new Color32(255, 255, 255, 255) : new Color32(0, 0, 0, 0);
        }
        _circleTex.SetPixels32(px);
        _circleTex.Apply(false, true);
        _circleSprite = Sprite.Create(_circleTex, new Rect(0, 0, s, s), new Vector2(0.5f, 0.5f), s / (ColliderRadius * 2f));
    }

    private void BuildCorridor()
    {
        if (!SplineContainer || SplineSampleCount < 2) return;

        var spline = SplineContainer.Spline;
        float totalLen = spline.GetLength();
        totalLen = Mathf.Max(totalLen, 1f);

        var trs = SplineContainer.transform;
        Vector2 prevPos = Vector2.zero;
        Vector2 prevTan = Vector2.right;
        for (int i = 0; i < SplineSampleCount; i++)
        {
            float u = i / (float)(SplineSampleCount - 1);
            float d = u * totalLen;
            spline.Evaluate(d, out var posL, out var tanL, out _);
            Vector3 posW = trs.TransformPoint((Vector3)posL);
            Vector3 tanW = trs.TransformDirection((Vector3)tanL);
            Vector2 p = new Vector2(posW.x, posW.y);
            Vector2 t = ((Vector2)tanW).normalized;
            if (i > 0)
            {
                float segLen = Vector2.Distance(prevPos, p);
                if (segLen > 1e-4f)
                {
                    Vector2 n = new Vector2(-t.y, t.x).normalized;
                    Vector2 mid = (prevPos + p) * 0.5f;

                    CreateWall(mid + n * (CorridorWidth * 0.5f), t, segLen);
                    CreateWall(mid - n * (CorridorWidth * 0.5f), t, segLen);
                }
            }
            prevPos = p;
            prevTan = t;
        }
    }

    private void CreateWall(Vector2 center, Vector2 tangent, float length)
    {
        var go = new GameObject("Wall");
        go.transform.position = center;
        go.transform.rotation = Quaternion.FromToRotation(Vector3.right, new Vector3(tangent.x, tangent.y, 0));
        var bc = go.AddComponent<BoxCollider2D>();
        bc.size = new Vector2(length, Mathf.Max(0.02f, WallThickness));
        bc.usedByComposite = false;
    }

    private void PrewarmPool()
    {
        for (int i = 0; i < Capacity; i++)
        {
            var go = new GameObject("Ball_" + i);
            var sr = go.AddComponent<SpriteRenderer>();
            sr.sprite = _circleSprite;
            var col = go.AddComponent<CircleCollider2D>();
            col.radius = ColliderRadius;
            col.sharedMaterial = _mat;
            var rb = go.AddComponent<Rigidbody2D>();
            rb.gravityScale = 1f; // use global gravity
            rb.drag = LinearDamping * 10f; // approximate
            rb.angularDrag = 0.05f;
            rb.interpolation = RigidbodyInterpolation2D.None;

            go.SetActive(false);
            _pool.Add(go);
            _bodies.Add(rb);
        }
    }

    private void PrimeSpawn()
    {
        int toSpawn = Mathf.Min(BurstOnStart > 0 ? BurstOnStart : (SpawnPerSecond == 0 ? MaxActive : 0), MaxActive);
        SpawnN(toSpawn);
    }

    private void SpawnN(int n)
    {
        if (SpawnPoint == null) return;
        int spawned = 0;
        Vector2 sp = new Vector2(SpawnPoint.position.x, SpawnPoint.position.y);
        for (int i = 0; i < _pool.Count && spawned < n; i++)
        {
            var go = _pool[i];
            if (go.activeSelf) continue;
            go.transform.position = sp;
            go.transform.rotation = Quaternion.identity;
            var rb = _bodies[i];
            Vector2 dir = new Vector2(
                Mathf.Lerp(vecA.x, vecA.y, (float)_rng.NextDouble()),
                Mathf.Lerp(vecB.x, vecB.y, (float)_rng.NextDouble())
            ).normalized;
            if (!float.IsFinite(dir.x) || !float.IsFinite(dir.y)) dir = Vector2.right;
            rb.velocity = dir * 6f;
            go.SetActive(true);
            _active++;
            spawned++;
            if (_active >= MaxActive) break;
        }
    }

    private void Update()
    {
        _swTotal.Restart();

        // spawn
        if (SpawnPerSecond > 0 && _active < MaxActive)
        {
            _spawnAcc += Mathf.Max(Time.deltaTime, 0f) * SpawnPerSecond;
            int spawnNow = (int)_spawnAcc;
            if (spawnNow > 0)
            {
                _spawnAcc -= spawnNow;
                SpawnN(spawnNow);
            }
        }

        // scripted physics simulate
        _accum += Time.deltaTime;
        double step = Time.fixedDeltaTime;
        while (_accum >= step)
        {
            _swSim.Restart();
            Physics2D.Simulate((float)step);
            _swSim.Stop();
            _sumSim += _swSim.Elapsed.TotalMilliseconds;
            _accum -= step;
        }

        _swTotal.Stop();
        _sumTotal += _swTotal.Elapsed.TotalMilliseconds;
        _frames++;

        if (BenchmarkEnabled && _frames % Math.Max(1, BenchmarkReportInterval) == 0)
        {
            double avgTotal = _sumTotal / _frames;
            double avgSim = _sumSim / _frames;
            Debug.Log($"[UNITY_PHYS2D] avg total {avgTotal:0.00} ms | simulate {avgSim:0.00} ms (frames={_frames})");
            _frames = 0; _sumTotal = 0; _sumSim = 0; _sumUpload = 0;
        }
    }
}
