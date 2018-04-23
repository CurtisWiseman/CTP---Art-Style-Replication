#include "UnityCG.cginc"
#include "UnityStandardConfig.cginc"
#include "UnityLightingCommon.cginc"

half4 BRDF_Unity_Toon(half3 diffColor, half3 specColor, half oneMinusReflectivity, half smoothness,
	half3 normal, half3 viewDir,
	UnityLight light, UnityIndirect gi)
{
	half3 halfDir = Unity_SafeNormalize(light.dir + viewDir);

	half nl = smoothstep(0.0, 0.05, saturate(dot(normal, light.dir)));
	half nh = saturate(dot(normal, halfDir));
	half nv = saturate(dot(normal, viewDir));
	half lh = saturate(dot(light.dir, halfDir));

	// Specular term
	half perceptualRoughness = SmoothnessToPerceptualRoughness(smoothness);
	half roughness = PerceptualRoughnessToRoughness(perceptualRoughness);

#if UNITY_BRDF_GGX

	half a = roughness;
	half a2 = a*a;

	half d = nh * nh * (a2 - 1.h) + 1.00001h;
#ifdef UNITY_COLORSPACE_GAMMA
	half specularTerm = a / (max(0.32h, lh) * (1.5h + roughness) * d);
#else
	half specularTerm = a2 / (max(0.1h, lh*lh) * (roughness + 0.5h) * (d * d) * 4);
#endif

#if defined (SHADER_API_MOBILE)
	specularTerm = specularTerm - 1e-4h;
#endif

#else

	// Legacy
	half specularPower = PerceptualRoughnessToSpecPower(perceptualRoughness);

	half invV = lh * lh * smoothness + perceptualRoughness * perceptualRoughness;
	half invF = lh;

	half specularTerm = ((specularPower + 1) * pow(nh, specularPower)) / (8 * invV * invF + 1e-4h);

#ifdef UNITY_COLORSPACE_GAMMA
	specularTerm = sqrt(max(1e-4h, specularTerm));
#endif

#endif

#if defined (SHADER_API_MOBILE)
	specularTerm = clamp(specularTerm, 0.0, 100.0); // Prevent FP16 overflow on mobiles
#endif
#if defined(_SPECULARHIGHLIGHTS_OFF)
	specularTerm = 0.0;
#endif

	// surfaceReduction = Int D(NdotH) * NdotH * Id(NdotL>0) dH = 1/(realRoughness^2+1)

	// 1-0.28*x^3 as approximation for (1/(x^4+1))^(1/2.2) on the domain [0;1]
	// 1-x^3*(0.6-0.08*x)   approximation for 1/(x^4+1)
#ifdef UNITY_COLORSPACE_GAMMA
	half surfaceReduction = 0.28;
#else
	half surfaceReduction = (0.6 - 0.08*perceptualRoughness);
#endif

	surfaceReduction = 1.0 - roughness*perceptualRoughness*surfaceReduction;

	half grazingTerm = saturate(smoothness + (1 - oneMinusReflectivity));
	half3 color = (diffColor + specularTerm * specColor) * light.color * nl
		+ gi.diffuse * diffColor
		+ surfaceReduction * gi.specular * FresnelLerpFast(specColor, grazingTerm, nv);

	return half4(color, 1);
}