using System.Xml;
using System.Collections.Generic;
using UnityEngine;

public class FileMonitorScreenshotter : MonoBehaviour
{
    private long ID_number = -1;
    
    [SerializeField] private List<int> effectsEnabled;
    [SerializeField] private List<Dictionary<string, float>> effectParameters;
    //private List<Material> effects;
    [SerializeField] private Renderer rend;
    private string filePath = "C:/Users/can0b/Desktop/Uni Work/CTP_Art_Style_Replication/UnityScreenshots/";
    [SerializeField] private TextAsset geneticOutput;
    XmlDocument geneticOutputXML = new XmlDocument();

	// Use this for initialization
	void Start ()
    {
        //effects = new List<Material>(rend.sharedMaterials);
        ReadFileInfo();
        ApplyInfoToShaders();
        TakeScreenshot();
    }

    // Update is called once per frame
    void Update()
    {
        if(ReadFileInfo())
        {
            ApplyInfoToShaders();
            TakeScreenshot();
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
            //Debug.Log("New ID number: " + temp_ID);
            //Debug.Log("Number of shaders in XML: " + shaders.Count);
            //Debug.Log("Number of effects in code (before): " + effectsEnabled.Count);

            if (effectsEnabled.Count != shaders.Count)
            {
                effectsEnabled = new List<int>(new int[shaders.Count]);
                effectParameters = new List<Dictionary<string, float>>(new Dictionary<string,float>[shaders.Count]);
            }

            //Debug.Log("Number of effects in code (after): " + effectsEnabled.Count);
            //Debug.Log("Number of parameter sets in code (after): " + effectParameters.Count);

            for (int i = 0; i < shaders.Count; i++)
            {
                XmlNodeList children = shaders[i].ChildNodes;
                effectsEnabled[i] = int.Parse(children[0].InnerText);
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
                rend.sharedMaterials[i].SetFloat("_" + param.Key, param.Value);
                Debug.Log("Effect " + i + " setting property _" + param.Key + " to " + param.Value);
                Debug.Log(param.Key + ": " + rend.sharedMaterials[i].GetFloat("_" + param.Key));
            }
        }
    }

    void TakeScreenshot()
    {
        ScreenCapture.CaptureScreenshot(filePath + ID_number.ToString() + ".png");
        Debug.Log("Saved screenshot: " + ID_number.ToString() + ".png");
    }
}