using System;
using UnityEngine;
using TMPro;

[System.Serializable]
public class NetworkConfiguration
{
    public string IPAddress;
    public string keyptPortNum;
    public string camPortNum;
    public string graphPortNum;
    public string resolutionPortNum;

    public string PausePortNum;

    public bool isIPAllocated ()
    {
        if (String.Equals(IPAddress, "undefined"))
            return false;
        else
            return true;
    }
}

public class NetworkManager : MonoBehaviour
{
    // Loading the Network Configurations
    public NetworkConfiguration netConfig;

    // Display variables for menu
    public TextMeshPro IPDisplay;

    // To indicate no IP
    private bool IPNotFound;

    public string getKeypointAddress()
    {
        if (IPNotFound)
            return "tcp://:";
        else
            return "tcp://" + netConfig.IPAddress + ":" + netConfig.keyptPortNum;
    }

    public string getCamAddress()
    {
        if (IPNotFound)
            return "tcp://:";
        else
            return "tcp://" + netConfig.IPAddress + ":" + netConfig.camPortNum;
    }

    public string getGraphAddress()
    {
        if (IPNotFound)
            return "tcp://:";
        else
            return "tcp://" + netConfig.IPAddress + ":" + netConfig.graphPortNum;
    }

    public string getResolutionAddress()
    {
        if (IPNotFound)
            return "tcp://:";
        else
            return "tcp://" + netConfig.IPAddress + ":" + netConfig.resolutionPortNum;

    }

    public string getPauseAddress()
    {
        if (IPNotFound)
            return "tcp://:";
        else
            return "tcp://" + netConfig.IPAddress + ":" + netConfig.PausePortNum;
        
    }


    public void changeIPAddress(string IPAddress)
    {
        netConfig.IPAddress = IPAddress;
        IPNotFound = false;

        // Storing in the Oculus Player Preferences Dict
        PlayerPrefs.SetString("ipAddress", IPAddress);
    }


    void Start()
    {
        var jsonFile = Resources.Load<TextAsset>("Configurations/Network");
        netConfig = JsonUtility.FromJson<NetworkConfiguration>(jsonFile.text);

        if (PlayerPrefs.HasKey("ipAddress"))
            netConfig.IPAddress = PlayerPrefs.GetString("ipAddress");

        if (!netConfig.isIPAllocated())
            IPNotFound = true;
        else
            IPNotFound = false;        
    }

    void Update()
    {
        // Displaying IP information
        if (!IPNotFound)
            IPDisplay.text = "IP Address: " + netConfig.IPAddress;
        else
            IPDisplay.text = "IP Address: Not Specified";
    }
}