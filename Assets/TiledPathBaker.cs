// using System;
// using System.Collections.Generic;
// using System.IO;
// using System.Linq;
// using UnityEngine;
//
// // Senin LevelLoader içindeki tipleri kullanıyoruz:
// using LevelObject = LevelLoader.Object;
// using TiledLayer   = LevelLoader.Layer;
// using TiledBoard   = LevelLoader.BoardData;
//
// public enum TiledObjectType
// {
//     None = 0,
//     PathNode = 1,
//     Spawner  = 2,
//     Obstacle = 3,
// }
//
// [Serializable]
// public class GidMapEntry
// {
//     public int gid;
//     public TiledObjectType type;
//     public float obstacleRadius = 0.5f; // obstacle ise kullanılacak yarıçap (dilersen obj.width/2 de yapabilirsin)
//     public bool  obstacleOpen   = false;
// }
//
// // Editörde kolay kurulum için:
// public class TiledPathBaker : MonoBehaviour
// {
//     [Header("Input (Tiled)")]
//     public TextAsset levelJson;           // /StreamingAssets’ten veya Resources’tan da okuyabilirsin
//     public float pixelsPerUnit = 64f;     // örn: 64px = 1u
//
//     [Tooltip("Path/Spawner/Obstacle için GID eşlemesi")]
//     public List<GidMapEntry> gidMap = new()
//     {
//         // ÖRNEK: GID 10 → PathNode, GID 12 → Obstacle, GID 13 → Spawner v.b.
//         new GidMapEntry{ gid = 10, type = TiledObjectType.PathNode },
//         new GidMapEntry{ gid = 12, type = TiledObjectType.Obstacle, obstacleRadius = 0.5f, obstacleOpen = false },
//         new GidMapEntry{ gid = 13, type = TiledObjectType.Spawner },
//         new GidMapEntry{ gid = 14, type = TiledObjectType.PathNode },
//     };
//
//     [Header("Path Resample")]
//     [Tooltip("Path polyline resample adımı (Unity units)")]
//     public float step = 0.50f;
//
//     [Header("Defaults for Physics")]
//     public float corridorWidth = 6f;
//     public float colliderRadius = 0.06f;
//     public float gridCellSize = 0.12f;
//
//     [Header("Output (JSON)")]
//     public string saveFileName = "BakedLevel.json";  // Assets/… altına kayıt için
// #if UNITY_EDITOR
//     [ContextMenu("Bake From Tiled JSON")]
//     private void BakeNow_Context()
//     {
//         if (levelJson == null)
//         {
//             Debug.LogError("[TiledPathBaker] levelJson atanmadı.");
//             return;
//         }
//
//         var baked = Bake(levelJson.text, pixelsPerUnit, gidMap, step, corridorWidth, colliderRadius, gridCellSize);
//         if (baked == null)
//         {
//             Debug.LogError("[TiledPathBaker] Bake başarısız.");
//             return;
//         }
//
//         // Kaydet
//         var json = JsonUtility.ToJson(baked, true);
//         var path = Path.Combine(Application.dataPath, saveFileName);
//         File.WriteAllText(path, json);
//         Debug.Log($"[TiledPathBaker] Kaydedildi: {path}");
//
//         // Unity’ye asset olarak import etmek istersen:
//         UnityEditor.AssetDatabase.Refresh();
//     }
// #endif
//
//     public static BakedLevel Bake(
//         string tiledJson, float ppu, List<GidMapEntry> map, float resampleStep,
//         float corridorWidth, float colliderRadius, float gridCellSize)
//     {
//         if (string.IsNullOrEmpty(tiledJson)) return null;
//
//         var board = JsonUtility.FromJson<TiledBoard>(tiledJson);
//         if (board == null || board.layers == null)
//         {
//             Debug.LogError("[TiledPathBaker] Board parse edilemedi.");
//             return null;
//         }
//
//         var baked = new BakedLevel
//         {
//             pixelsPerUnit = Mathf.Max(1e-3f, ppu),
//             tileWidthPx = board.tilewidth,
//             tileHeightPx = board.tileheight,
//             mapWidthTiles = board.width,
//             mapHeightTiles = board.height,
//             corridorWidth = corridorWidth,
//             colliderRadius = colliderRadius,
//             gridCellSize = gridCellSize,
//             sourceLevelName = "TiledLevel"
//         };
//
//         // Tiled koordinatlarını Unity’ye çevir: (0,0) sol-üst → (0,0) dünya merkezi olacak şekilde
//         // Varsayılan çeviri: dünya orijini haritanın merkezine konur.
//         float mapWpx = board.width  * board.tilewidth;
//         float mapHpx = board.height * board.tileheight;
//         Vector2 worldOrigin = new Vector2(-mapWpx * 0.5f, +mapHpx * 0.5f); // Tiled y aşağı artar → +H/2 üstte
//         float invPPU = 1f / baked.pixelsPerUnit;
//
//         // GID → tip lookup
//         var gidLookup = new Dictionary<int, GidMapEntry>();
//         foreach (var e in map) gidLookup[e.gid] = e;
//
//         // Object layer’ları gez
//         List<Vector2> pathNodes = new();
//         int spawnerAutoId = 0;
//
//         foreach (var layer in board.layers)
//         {
//             if (layer.type != "objectgroup" || layer.objects == null) continue;
//
//             foreach (LevelObject o in layer.objects)
//             {
//                 if (o == null) continue;
//                 if (!gidLookup.TryGetValue(o.gid, out var entry)) continue;
//
//                 // Tiled obj koordinatı: (x,y) sol-üstten piksel; Unity’ye çevir:
//                 // Tiled’de objenin referans noktası (varsayılan) sol-üst köşe.
//                 float px = o.x + o.width * 0.5f;
//                 float py = o.y + o.height * 0.5f;
//                 // Dünya uzayı (birim):
//                 float wx = (worldOrigin.x + px) * invPPU;
//                 float wy = (worldOrigin.y - py) * invPPU; // y ekseni ters
//
//                 switch (entry.type)
//                 {
//                     case TiledObjectType.PathNode:
//                         pathNodes.Add(new Vector2(wx, wy));
//                         break;
//
//                     case TiledObjectType.Spawner:
//                         baked.spawners.Add(new BakedLevel.Spawner { x = wx, y = wy, id = spawnerAutoId++ });
//                         break;
//
//                     case TiledObjectType.Obstacle:
//                         float r = entry.obstacleRadius;
//                         if (o.width > 0) r = Mathf.Max(r, (o.width * 0.5f) * invPPU);
//                         baked.obstacles.Add(new BakedLevel.Obstacle
//                         {
//                             x = wx, y = wy, radius = r, open = entry.obstacleOpen, id = o.id
//                         });
//                         break;
//                 }
//             }
//         }
//
//         if (pathNodes.Count == 0)
//         {
//             Debug.LogWarning("[TiledPathBaker] Path node bulunamadı (GID eşleşmesi?).");
//         }
//         else
//         {
//             // Path’i sıralama: en basiti x’e göre; dilersen objeye 'order' property koyup ona göre sırala.
//             pathNodes = pathNodes
//                 .OrderBy(v => v.x)
//                 .ThenBy(v => -v.y)
//                 .ToList();
//
//             // Yeniden örnekleme (uniform step)
//             var resampled = ResamplePolyline(pathNodes, Mathf.Max(0.02f, resampleStep));
//             foreach (var p in resampled)
//                 baked.path.Add(new BakedLevel.PathPoint { x = p.x, y = p.y });
//         }
//
//         return baked;
//     }
//
//     // Basit polyline resampler (Catmull-Rom yok; segmentler boyunca doğrusal örnekler)
//     public static List<Vector2> ResamplePolyline(List<Vector2> pts, float step)
//     {
//         var outPts = new List<Vector2>();
//         if (pts == null || pts.Count == 0) return outPts;
//         if (pts.Count == 1) { outPts.Add(pts[0]); return outPts; }
//
//         outPts.Add(pts[0]);
//         float acc = 0f;
//
//         for (int i = 0; i < pts.Count - 1; i++)
//         {
//             var a = pts[i];
//             var b = pts[i + 1];
//             float len = Vector2.Distance(a, b);
//             if (len < 1e-5f) continue;
//
//             float t = step - acc;
//             while (t <= len)
//             {
//                 var p = Vector2.Lerp(a, b, t / len);
//                 outPts.Add(p);
//                 t += step;
//             }
//             acc = (acc + len) % step;
//         }
//
//         // Son noktayı garanti et
//         if (outPts.Count == 0 || (outPts[outPts.Count - 1] - pts[^1]).sqrMagnitude > 1e-6f)
//             outPts.Add(pts[^1]);
//
//         return outPts;
//     }
// }
