Shader "Unlit/InstancedSpriteArray_Builtin"
{
    Properties
    {
        _BaseColor ("Tint", Color) = (1,1,1,1)
        _SpriteArray ("Sprite Array", 2DArray) = "" {}

        // Z-order kontrol
        _ZBase   ("Z Base", Float) = 0
        _ZScale  ("Z Scale per Unit", Float) = -0.001   // sortKey ile çarpılır
        _ZWrite  ("ZWrite (0/1)", Float) = 0
        _ZJitter ("Z Jitter", Float) = 0                // opsiyonel
    }
    SubShader
    {
        Tags { "Queue"="Transparent" "RenderType"="Transparent" }
        Cull Off
        ZTest LEqual
        ZWrite [_ZWrite]
        Blend SrcAlpha OneMinusSrcAlpha

        Pass
        {
            CGPROGRAM
            #pragma target 4.5
            #pragma require 2darray
            #pragma vertex   vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            UNITY_DECLARE_TEX2DARRAY(_SpriteArray);
            float4 _BaseColor;
            float  _ZBase, _ZScale, _ZJitter;

            // X,Y,R,Angle | SpriteIndex | SortKey
            struct ParticleGPU { float4 xyra; float spriteIndex; float sortKey; };
            StructuredBuffer<ParticleGPU> _Particles;

            struct appdata { float3 vertex:POSITION; float2 uv:TEXCOORD0; uint id:SV_InstanceID; };
            struct v2f { float4 pos:SV_Position; float2 uv:TEXCOORD0; nointerpolation uint slice:TEXCOORD1; };

            float2 rot2(float2 v, float a){ float s=sin(a), c=cos(a); return float2(c*v.x - s*v.y, s*v.x + c*v.y); }

            v2f vert (appdata v)
            {
                v2f o;
                ParticleGPU p = _Particles[v.id];

                float2 center = p.xyra.xy;
                float  r      = p.xyra.z;
                float  ang    = p.xyra.w;
                uint   slice  = (uint)p.spriteIndex;

                // Yaş/ID bazlı Z: sortKey (C#) * _ZScale
                float z = _ZBase + p.sortKey * _ZScale;

                // opsiyonel ufak jitter (aynı sortKey’de yapışmayı kırmak için)
                if (_ZJitter > 0)
                {
                    float h = frac(sin((v.id + 1) * 12.9898) * 43758.5453);
                    z += h * _ZJitter;
                }

                float2 world2 = center + rot2(v.vertex.xy * r, ang);
                o.pos  = mul(UNITY_MATRIX_VP, float4(world2, z, 1));
                o.uv   = v.uv;
                o.slice= slice;
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                fixed4 c = UNITY_SAMPLE_TEX2DARRAY(_SpriteArray, float3(i.uv, i.slice));
                return c * _BaseColor;
            }
            ENDCG
        }
    }
    Fallback Off
}
