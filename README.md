# 2D Object Fall Simulation (Unity)

A compact setup for testing large-scale 2D falling object simulations using Unity Jobs + Burst, along with a comparison scene using Unity Physics2D.

## Contents
- **ObjectFallSimulation.cs**: Custom particle-based physics (Jobs/Burst), path following, spatial grid, obstacles, GPU instanced rendering.
- **UnityPhysics2D_Bench.cs**: Baseline benchmark using Rigidbody2D and BoxCollider2D corridor walls.
- **TiledPathBaker.cs** + **BakeLevel.cs**: Converts Tiled JSON into a baked level (path, spawners, obstacles).  
- **BakedLevel_Demo.json**: Example baked level.

## Workflow
1. Author level in **Tiled**.
2. Bake it using **TiledPathBaker** to generate a `BakedLevel` JSON.
3. Load the baked file in **ObjectFallSimulation**.
4. Run or compare with the **Physics2D benchmark** scene.

## Features
- Path-based corridor movement.
- Particle collisions via grid-based broadphase.
- Circular obstacles.
- Instanced rendering for thousands of objects.
- Optional adaptive gravity and sleep logic.

## Requirements
- Unity with Burst, Jobs, Collections, Mathematics.
- Splines package (for Physics2D benchmark).
