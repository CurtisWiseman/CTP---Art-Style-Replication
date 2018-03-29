Shader "Custom/SpriteShader" {
	Properties{
		_MainTex("Base (RGB)", 2D) = "white" {}
	_Bump("Bump", 2D) = "bump" {}
	_Tooniness("Tooniness", Range(0.1,20)) = 4
		_ColorMerge("Color Merge", Range(0.1,20000)) = 8
		_Ramp("Ramp Texture", 2D) = "white" {}
	}
		SubShader{
		Tags{ "RenderType" = "Opaque" }
		LOD 200

		CGPROGRAM
		// Upgrade NOTE: excluded shader from Xbox360 because it uses wrong array syntax (type[size] name)
#pragma exclude_renderers xbox360
#pragma surface surf Toon

		sampler2D _MainTex;
	sampler2D _Bump;
	sampler2D _Ramp;
	float _Tooniness;
	float _ColorMerge;

	struct Input {
		float2 uv_MainTex;
		float2 uv_Bump;
	};

	void surf(Input IN, inout SurfaceOutput o) {
		float4 c = tex2D(_MainTex, IN.uv_MainTex);

		o.Normal = UnpackNormal(tex2D(_Bump, IN.uv_Bump));
		o.Albedo = floor(c.rgb*_ColorMerge) / _ColorMerge;
		o.Alpha = c.a;
	}

	half4 LightingToon(SurfaceOutput s, half3 lightDir, half3 viewDir, half atten)
	{
		half4 c;
		half NdotL = dot(s.Normal, lightDir);
		NdotL = tex2D(_Ramp, float2(NdotL, 0.5));

		half outline = dot(s.Normal, viewDir);

		if (outline <= 0.1) { c.rgb = 0; }
		else {

			c.rgb = s.Albedo * _LightColor0.rgb * NdotL * atten * 2;
		}

		c.a = s.Alpha;
		return c;
	}

	ENDCG
	}
		FallBack "Diffuse"
}