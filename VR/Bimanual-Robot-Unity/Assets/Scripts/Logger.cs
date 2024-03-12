using UnityEngine.UI;
using System.Collections.Generic;
using UnityEngine;

class Logger
{
    private static Text Text;

    //List of strings 
    private static List<string> Logs = new List<string>();

    // Limit on the number of strings that can be present in the log at a time
    public static int LogLimit = 5;


    // Constructor
    static Logger()
    {
        // Obtaining the OVR Object
        OVRCameraRig OVRCamera = GameObject.FindObjectOfType<OVRCameraRig>();

        // Creating the Canvas to print the text
        GameObject goCanvas = new GameObject("Canvas");
        goCanvas.transform.parent = OVRCamera.gameObject.transform;
        goCanvas.transform.position = new Vector3(0, -8, 0); 
        Canvas canvas = goCanvas.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.WorldSpace;

        // Adding a text component to render the logs
        GameObject goText = new GameObject("Text");
        goText.transform.parent = goCanvas.transform;
        goText.transform.position = new Vector3(0, 0, 10);
        goText.transform.rotation = new Quaternion(0, 0, 0, 0);

        //Formating the text
        Text = goText.AddComponent<Text>();
        Text.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
        Text.fontSize = 30;
        Text.color = Color.black;

        // Performing other transforms on the text box
        RectTransform tr = goText.GetComponent<RectTransform>();
        tr.localScale = new Vector3(0.01f, 0.01f, 0.01f);
        tr.sizeDelta = new Vector2(1000, 1000);
    }


    // Function to generate the string to be logged
    private static void PrintLogs()
    {
        string StringToLog = "";
        foreach (string item in Logs)
            StringToLog += item + "\n";

        Text.text = StringToLog;
    }


    // Logging function
    public static void Log(string log)
    {
        string StringToLog = log.ToString();
        Logs.Add(StringToLog);

        // Removing the last string
        if (Logs.Count > LogLimit)
            Logs.RemoveAt(0);

        PrintLogs();
    }
}