using System;
using System.Collections.Generic;
using UnityEngine;

[Serializable]
public class BakedLevel
{
    [Serializable]
    public struct PathPoint { public float x, y; }

    [Serializable]
    public struct Spawner { public float x, y; public int id; }

    [Serializable]
    public struct Obstacle { public float x, y; public float radius; public bool open; public int id; }

    // Ana veriler
    public List<PathPoint> path = new();
    public List<Spawner> spawners = new();
    public List<Obstacle> obstacles = new();

    // Fizik grid meta
    public float corridorWidth = 6f;
    public float colliderRadius = 0.06f;
    public float gridCellSize = 0.12f;

    // Tiled → Unity dönüşüm meta (debug/geri-izleme amaçlı)
    public int tileWidthPx;
    public int tileHeightPx;
    public int mapWidthTiles;
    public int mapHeightTiles;
    public float pixelsPerUnit; // 64px = 1u için 64

    // Kaynak dosya adı (opsiyonel)
    public string sourceLevelName;

    // Yardımcı
    public Vector2[] GetPathArray()
    {
        var a = new Vector2[path.Count];
        for (int i = 0; i < a.Length; i++) a[i] = new Vector2(path[i].x, path[i].y);
        return a;
    }
}

// Çok ufak bir yardımcı: TextAsset → BakedLevel
public static class BakedLevelLoader
{
    public static BakedLevel FromTextAsset(TextAsset json)
    {
        if (json == null || string.IsNullOrEmpty(json.text))
            return null;
        return JsonUtility.FromJson<BakedLevel>(json.text);
    }

    public static TextAsset ToTextAsset(BakedLevel baked, string name = "BakedLevel")
    {
        string s = JsonUtility.ToJson(baked, true);
        var ta = new TextAsset(s);
        ta.name = name;
        return ta;
    }
}