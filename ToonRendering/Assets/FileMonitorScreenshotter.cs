using System.Xml;
using System.Collections.Generic;
using UnityEngine;

public class FileMonitorScreenshotter : MonoBehaviour
{
    private long ID_number = -1;

    [SerializeField] private Camera screenCam;
    [SerializeField] private List<int> effectsEnabled;
    [SerializeField] private List<Dictionary<string, float>> effectParameters;
    //private List<Material> effects;
    [SerializeField] private Renderer rend;
    private string filePath = "../UnityScreenshots/";
    [SerializeField] private TextAsset geneticOutput;
    XmlDocument geneticOutputXML = new XmlDocument();

	// Use this for initialization
	void Start ()
    {
        //effects = new List<Material>(rend.sharedMaterials);
        ReadFileInfo();
        ApplyInfoToShaders();
        TakeBetterScreenshot();
    }

    // Update is called once per frame
    void Update()
    {
        bool changed = ReadFileInfo();
        if (changed)
        {
            ApplyInfoToShaders();
            TakeBetterScreenshot();
        }
    }

    bool ReadFileInfo()
    {
        geneticOutputXML.LoadXml(geneticOutput.text);

        long temp_ID = long.Parse(geneticOutputXML.GetElementsByTagName("IDNumber").Item(0).InnerText);

        if (temp_ID > ID_number)
        {
            ID_number = temp_ID;
            XmlNodeList shaders = geneticOutputXML.GetElementsByTagName("shader");

            if (effectsEnabled.Count != shaders.Count)
            {
                effectsEnabled = new List<int>(new int[shaders.Count]);
                effectParameters = new List<Dictionary<string, float>>(new Dictionary<string,float>[shaders.Count]);
            }


            for (int i = 0; i < shaders.Count; i++)
            {
                XmlNodeList children = shaders[i].ChildNodes;
                effectsEnabled[i] = (int)float.Parse(children[0].InnerText);
                effectParameters[i] = new Dictionary<string, float>();

                for (int j = 0; j < (children.Count - 1); j++)
                {
                    string name = children[j + 1].Name;
                    string innertext = children[j + 1].InnerText;
                    effectParameters[i].Add(name, float.Parse(innertext));
                }
            }
            return true;
        }
        return false;
    }

    void ApplyInfoToShaders()
    {
        for (int i = 0; i < effectsEnabled.Count; i++)
        {
            rend.sharedMaterials[i].SetInt(Shader.PropertyToID("Enabled"), effectsEnabled[i]);
            Debug.Log(rend.sharedMaterials[i].shader.name);

            foreach (KeyValuePair<string,float> param in effectParameters[i])
            {
                rend.sharedMaterials[i].SetFloat(param.Key, param.Value);
                Debug.Log("Effect " + i + " setting property " + param.Key + " to " + param.Value);
                Debug.Log(param.Key + ": " + rend.sharedMaterials[i].GetFloat(param.Key));
            }
        }
    }

    void TakeScreenshot()
    {
        ScreenCapture.CaptureScreenshot(filePath + ID_number.ToString() + ".png");
        Debug.Log("Saved screenshot: " + ID_number.ToString() + ".png");
    }

    void TakeBetterScreenshot()
    {
        int resWidth = 650;
        int resHeight = 450;
        RenderTexture rt = new RenderTexture(resWidth, resHeight, 24);
        screenCam.targetTexture = rt;
        Texture2D screenShot = new Texture2D(resWidth, resHeight, TextureFormat.RGB24, false);
        screenCam.Render();
        RenderTexture.active = rt;
        screenShot.ReadPixels(new Rect(0, 0, resWidth, resHeight), 0, 0);
        screenCam.targetTexture = null;
        RenderTexture.active = null; // JC: added to avoid errors
        Destroy(rt);
        byte[] bytes = screenShot.EncodeToPNG();
        string filename = filePath + ID_number.ToString() + ".png";
        System.IO.File.WriteAllBytes(filename, bytes);
        Debug.Log("Saved screenshot: " + ID_number.ToString() + ".png");
    }
}