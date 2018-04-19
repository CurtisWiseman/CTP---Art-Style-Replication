Shader "Outline"
{
	Properties
	{
		_Color("Main Color", Color) = (0.5,0.5,0.5,1)
		_OutlineColor("Outline Color", Color) = (0,0,0,1)
		_ColourRed("ColourRed", Range(0.0, 1.0)) = 1.0
		_ColourGreen("_ColourGreen", Range(0.0, 1.0)) = 0.0
		_ColourBlue("_ColourBlue", Range(0.0, 1.0)) = 0.0
		_Width("Width", Range(0.0, 10.0)) = 0.03
		_MainTex("Base (RGB)", 2D) = "white" { }
	}

	CGINCLUDE

	#include "UnityCG.cginc"

	struct appdata
	{
		float4 vertex : POSITION;
		float3 normal : NORMAL;
	};

	struct v2f
	{
		float4 pos : POSITION;
		float4 color : COLOR;
	};

	uniform float _Width;
	uniform float _ColourRed;
	uniform float _ColourGreen;
	uniform float _ColourBlue;
	uniform float4 _OutlineColor;

	v2f vert(appdata v)
	{
		// just make a copy of incoming vertex data but scaled according to normal direction
		v2f o;
		o.pos = UnityObjectToClipPos(v.vertex);

		float3 norm = mul((float3x3)UNITY_MATRIX_IT_MV, v.normal);
		float2 offset = TransformViewToProjection(norm.xy);

		o.pos.xy += offset * o.pos.z * _Width;
		o.color = _OutlineColor;
		return o;
	}

	ENDCG

	SubShader
	{
		Tags{ "Queue" = "Transparent"}

		Pass
		{
			Name "OUTLINE"
			Tags{ "LightMode" = "Always" }

			Cull Off
			ZWrite Off
			ZTest Always
			ColorMask RGB

			// you can choose what kind of blending mode you want for the outline
			Blend SrcAlpha OneMinusSrcAlpha //Blend One OneMinusSrcAlpha	// Normal
											//Blend One One					// Additive
											//Blend One OneMinusDstColor	// Soft Additive
											//Blend DstColor Zero			// Multiplicative
											//Blend DstColor SrcColor		// 2x Multiplicative

			CGPROGRAM

			#pragma vertex vert
			#pragma fragment frag

			half4 frag(v2f i) :COLOR
			{
				return i.color;
			}

			ENDCG
		}
	}

	Fallback "Diffuse"
}