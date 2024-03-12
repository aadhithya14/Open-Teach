using System;
using UnityEngine;
using TMPro;

[System.Serializable]
public class NetworkConfiguration
{
    public string IPAddress;
    public string rightkeyptPortNum;

    public string leftkeyptPortNum;
    public string camPortNum;
    public string graphPortNum;
    public string resolutionPortNum;

    public string PausePortNum;

    public string rightgripperPortNum;

    public string leftgripperPortNum;

    public string LeftPausePortNum;

    public string RightPausePortNum;

    public string LeftGripperRotatePortNum;

    public string RightGripperRotatePortNum;

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

    public string getRightKeypointAddress()
    {
        if (IPNotFound)
            return "tcp://:";
        else
            return "tcp://" + netConfig.IPAddress + ":" + netConfig.rightkeyptPortNum;
    }

    public string getLeftKeypointAddress()
    {
        if (IPNotFound)
            return "tcp://:";
        else
            return "tcp://" + netConfig.IPAddress + ":" + netConfig.leftkeyptPortNum;
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

    public string getRightGripperAddress()
    {
         if (IPNotFound)
            return "tcp://:";
        else
            return "tcp://" + netConfig.IPAddress + ":" + netConfig.rightgripperPortNum;
    
    }

    public string getLeftGripperAddress()
    {
         if (IPNotFound)
            return "tcp://:";
        else
            return "tcp://" + netConfig.IPAddress + ":" + netConfig.leftgripperPortNum;
    
    }

    public string getLeftPauseStatus()
    {
        if (IPNotFound)
            return "tcp://:";
        else
            return "tcp://" + netConfig.IPAddress + ":" + netConfig.LeftPausePortNum;
    }

    public string getRightPauseStatus()
    {
        if (IPNotFound)
            return "tcp://:";
        else
            return "tcp://" + netConfig.IPAddress + ":" + netConfig.RightPausePortNum;
    }

     public string getLeftGripperRotateStatus()
    {
        if (IPNotFound)
            return "tcp://:";
        else
            return "tcp://" + netConfig.IPAddress + ":" + netConfig.LeftGripperRotatePortNum;
    }

    public string getRightGripperRotateStatus()
    {
        if (IPNotFound)
            return "tcp://:";
        else
            return "tcp://" + netConfig.IPAddress + ":" + netConfig.RightGripperRotatePortNum;
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