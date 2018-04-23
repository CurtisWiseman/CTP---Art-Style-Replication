Shader "Outline"
{
	Properties
	{
		_Color("Main Color", Color) = (0.5,0.5,0.5,1)
		_OutlineColor("Outline Color", Color) = (0,0,0,1)
		_O_ColourRed("ColourRed", Range(0.0, 1.0)) = 1.0
		_O_ColourGreen("ColourGreen", Range(0.0, 1.0)) = 0.0
		_O_ColourBlue("ColourBlue", Range(0.0, 1.0)) = 0.0
		_O_Width("Width", Range(0.0, 10.0)) = 0.03
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

	uniform float _O_Width;
	uniform float _O_ColourRed;
	uniform float _O_ColourGreen;
	uniform float _O_ColourBlue;
	uniform float4 _OutlineColor;

	v2f vert(appdata v)
	{
		// just make a copy of incoming vertex data but scaled according to normal direction
		v2f o;
		o.pos = UnityObjectToClipPos(v.vertex);

		float3 norm = mul((float3x3)UNITY_MATRIX_IT_MV, v.normal);
		float2 offset = TransformViewToProjection(norm.xy);

		o.pos.xy += offset * o.pos.z * _O_Width;
		o.color = float4(_O_ColourRed, _O_ColourGreen, _O_ColourBlue, 1);

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