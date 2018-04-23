Shader "Cel"
{
	Properties
	{
		_C_ColourRed("ColourRed", Range(0,1)) = 1
		_C_ColourGreen("ColourGreen", Range(0,1)) = 1
		_C_ColourBlue("ColourBlue", Range(0,1)) = 1
		_C_Levels("Levels", Range(0,1)) = 0.5
		_MainTex("Albedo (RGB)", 2D) = "white" {}
	}

	SubShader
	{
		Tags
		{
			"RenderType" = "Transparent"
			"Queue" = "AlphaTest+52"
		}
		LOD 200

		CGPROGRAM

		#pragma surface surf CelShadingForward
		#pragma target 3.0

		half4 LightingCelShadingForward(SurfaceOutput s, half3 lightDir, half atten) 
		{
			half NdotL = dot(s.Normal, lightDir);
			NdotL = 1 + clamp(floor(NdotL), -1, 0);
			NdotL = smoothstep(0, 0.025f, NdotL);
			half4 c;
			c.rgb = s.Albedo * _LightColor0.rgb * (NdotL * atten * 2);
			c.a = s.Alpha;
			return c;
		}

		sampler2D _MainTex;
		float _C_ColourRed;
		float _C_ColourGreen;
		float _C_ColourBlue;

		struct Input
		{
			float2 uv_MainTex;
		};

		void surf(Input IN, inout SurfaceOutput o)
		{
			fixed4 c = tex2D(_MainTex, IN.uv_MainTex) * float4(_C_ColourRed, _C_ColourGreen, _C_ColourBlue, 1);
			o.Albedo = c.rgb;
			o.Alpha = c.a;
		}
		ENDCG
	}
	FallBack "Diffuse"
}