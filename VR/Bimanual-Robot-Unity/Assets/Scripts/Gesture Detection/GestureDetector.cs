using System;
using System.Collections.Generic;

using UnityEngine;
using UnityEngine.UI;

using NetMQ;
using NetMQ.Sockets;

class GestureDetector : MonoBehaviour
{
    // Hand objects
    public OVRHand LeftHand;
    public OVRHand RightHand;
    public OVRSkeleton LeftHandSkeleton;
    public OVRSkeleton RightHandSkeleton;
    public OVRPassthroughLayer PassthroughLayerManager;
    private List<OVRBone> RightHandFingerBones;
    private List<OVRBone> LeftHandFingerBones;
    // Menu and RayCaster GameObjects
    public GameObject MenuButton;
    public GameObject ResolutionButton;
    public GameObject HighResolutionButton;
    public GameObject LowResolutionButton;
    // Useful UI tools for Wrist Tracking and Laser Pointer
    public GameObject WristTracker;
    private GameObject LaserPointer;
    private LineRenderer LineRenderer;
    // Hand Usage indicator
    public RawImage StreamBorder;
    // Stream Enablers
    bool StreamRelativeData = true;
    bool StreamAbsoluteRightData = false;
    bool StreamAbsoluteLeftData = false;
    bool SetRightGripperState = false;
    bool StreamResolution= true;
    public HighResolutionButtonController HighResolutionButtonController;
    public LowResolutionButtonController LowResolutionButtonController;
    // Network enablers
    private NetworkManager netConfig;
    private PushSocket rightclient;
    private PushSocket leftclient;
    private PushSocket client2;
    private string rightcommunicationAddress;
    private string leftcommunicationAddress;
    private string ResolutionAddress;
    private string state;
    private bool connectionEstablished = false;
    private bool leftconnectionEstablished = false;
    private bool rightconnectionEstablished = false;
    private bool resolutionconnectionEstablished = false;
    private bool PauseEstablished = false;
    private bool resolutioncreated = false;
    private bool IsRightGripper = false;
    private bool IsLeftGripper = false;
    private bool PauseRight = false;
    private bool PauseLeft = false;
    // Starting the server connection
    public void CreateTCPConnection()
     {
        // Check if communication address is available
        rightcommunicationAddress = netConfig.getRightKeypointAddress();
        bool RightAddressAvailable = !String.Equals(rightcommunicationAddress, "tcp://:");
        leftcommunicationAddress = netConfig.getLeftKeypointAddress();
        bool LeftAddressAvailable = !String.Equals(leftcommunicationAddress,  "tcp://:" );
        if (RightAddressAvailable)
        {
            // Initiate Push Socket
            rightclient = new PushSocket();
            rightclient.Connect(rightcommunicationAddress);
            rightconnectionEstablished = true;
        }
        if (LeftAddressAvailable)
        {
            leftclient = new PushSocket();
            leftclient.Connect(leftcommunicationAddress);
            leftconnectionEstablished = true;
        }
        // Setting color to green to indicate control
        if (rightconnectionEstablished && leftconnectionEstablished)
        {
            StreamBorder.color = Color.green;
            ToggleMenuButton(false);
        } else
        {
            StreamBorder.color = Color.red;
            ToggleMenuButton(true);
        }
    }
    // Function to toggle the menu button
    public void ToggleMenuButton(bool toggle)
    {
        MenuButton.SetActive(toggle);
        LineRenderer.enabled = toggle;
    }
    // Function to toggle the resolution button
    public void ToggleResolutionButton(bool toggle)
    {
        ResolutionButton.SetActive(toggle);
        LineRenderer.enabled = toggle;
    }
    // Function to toggle the high resolution button
    public void ToggleHighResolutionButton(bool toggle)
    {
        HighResolutionButton.SetActive(toggle);
        
    }
    // Function to toggle the low resolution button
    public void ToggleLowResolutionButton(bool toggle)
    {
        LowResolutionButton.SetActive(toggle);
    }
    
    // Start function
    void Start()
     {
        // Getting the Network Config Updater gameobject
        GameObject netConfGameObject = GameObject.Find("NetworkConfigsLoader");
        netConfig = netConfGameObject.GetComponent<NetworkManager>();
        // Getting the Laser Pointer and Line Renderer
        LaserPointer = GameObject.Find("LaserPointer");
        LineRenderer = LaserPointer.GetComponent<LineRenderer>();
        // Initializing the hand skeleton
        RightHandFingerBones = new List<OVRBone>(RightHandSkeleton.Bones);
        LeftHandFingerBones = new List<OVRBone>(LeftHandSkeleton.Bones);   
    }

    // Function to serialize the Vector3 List
    public static string SerializeVector3List(List<Vector3> gestureData)
    {
        string vectorString = "";
        foreach (Vector3 vec in gestureData)
            vectorString = vectorString + vec.x + "," + vec.y + "," + vec.z + "|";
        // Clipping last element and using a semi colon instead
        if (vectorString.Length > 0)
            vectorString = vectorString.Substring(0, vectorString.Length - 1) + ":";

        return vectorString;
    }

    public void SendRightHandData(String TypeMarker)
    {
        // Getting bone positional information
        List<Vector3> rightHandGestureData = new List<Vector3>();
        foreach (var bone in RightHandFingerBones)
        {
            Vector3 bonePositionright = bone.Transform.position;
            rightHandGestureData.Add(bonePositionright);
        }

        // Creating a string from the vectors
        string RightHandDataString = SerializeVector3List(rightHandGestureData);
        RightHandDataString = TypeMarker + ":" + RightHandDataString;

        rightclient.SendFrame(RightHandDataString);
    }
   
    public void SendLeftHandData(String TypeMarker)
    {
        // Getting bone positional information for left hand
        List<Vector3> leftHandGestureData = new List<Vector3>();
        foreach (var boneleft in LeftHandFingerBones)
        {
            Vector3 bonePositionleft = boneleft.Transform.position;
            leftHandGestureData.Add(bonePositionleft);
        }

        // Creating a string from the vectors
        string LeftHandDataString = SerializeVector3List(leftHandGestureData);
        LeftHandDataString = TypeMarker + ":" + LeftHandDataString;

        leftclient.SendFrame(LeftHandDataString);
    }

    public void SendResolution()
    {
        ResolutionAddress = netConfig.getResolutionAddress();
        bool Available = !String.Equals(ResolutionAddress, "tcp://:");

        if (Available)
        {   if (!resolutioncreated)
            // Initiate Push Socket
            { 
                Debug.Log("Address Available");
                client2 = new PushSocket();
                client2.Connect(ResolutionAddress);
                resolutionconnectionEstablished = true;
                resolutioncreated=true;
            }
            else 
            {
                resolutionconnectionEstablished=true;
            }
        }
        else
        {
            resolutionconnectionEstablished = false;
        }
        
        if (resolutionconnectionEstablished)
        {
            if (HighResolutionButtonController.HighResolution)
            {   
                state="High";
                client2.SendFrame(state);
                Debug.Log("High Button was clicked!");
            }

            else if (LowResolutionButtonController.LowResolution)
            {   
                state="Low";
                client2.SendFrame(state);
                Debug.Log("Low Button was clicked!");   
            }

            else 
            {   
                client2.SendFrame("None"); 
                Debug.Log("No button was pressed");

            }
        }
        else
        {
            client2.SendFrame("None");
        }
    }

   
    
    public void StreamPauser()
    {
        // Pausing the stream
        if (LeftHand.GetFingerIsPinching(OVRHand.HandFinger.Middle))
        {
            StreamRelativeData = false;
            StreamAbsoluteLeftData = true;
            StreamAbsoluteRightData = true;
            IsLeftGripper = !IsLeftGripper;
            StreamResolution= false;
            StreamBorder.color = Color.red; 
            ToggleMenuButton(false);
            ToggleResolutionButton(false);
            WristTracker.SetActive(true);
        }

        if (RightHand.GetFingerIsPinching(OVRHand.HandFinger.Middle))
        {
            StreamRelativeData = false;
            StreamAbsoluteRightData = true;
            StreamAbsoluteLeftData = true;
            IsRightGripper = !IsRightGripper;
            StreamResolution= false;
            StreamBorder.color = Color.red; 
            ToggleMenuButton(false);
            ToggleResolutionButton(false);
            WristTracker.SetActive(true); 
        }

        // Pause the stream
        if (LeftHand.GetFingerIsPinching(OVRHand.HandFinger.Ring))
        {
            StreamRelativeData = false;
            StreamAbsoluteLeftData = true;
            StreamAbsoluteRightData = true;
            PauseLeft= true;
            PauseRight= false;
            StreamResolution = false;
            StreamBorder.color = Color.red; // Green for right hand stream
            ToggleMenuButton(false);
            ToggleResolutionButton(false);
            WristTracker.SetActive(true);
        }

        if (RightHand.GetFingerIsPinching(OVRHand.HandFinger.Ring))
        {
            StreamRelativeData = false;
            StreamAbsoluteRightData = true;
            StreamAbsoluteLeftData = true;
            PauseRight= true;
            PauseLeft= false;
            StreamResolution = false;
            StreamBorder.color = Color.red; // Green for right hand stream
            ToggleMenuButton(false);
            ToggleResolutionButton(false);
            WristTracker.SetActive(true);
        }

        // Starting Stream
        if (LeftHand.GetFingerIsPinching(OVRHand.HandFinger.Index))
        {
            StreamRelativeData = false;
            StreamAbsoluteRightData = true;
            StreamAbsoluteLeftData = true;
            StreamResolution = false;
            PauseLeft= false;
            PauseRight= false;
            StreamBorder.color = Color.green; // Red color for no stream 
            ToggleMenuButton(false);
            ToggleResolutionButton(false);
            WristTracker.SetActive(true);  
        }
        // Gripper detection
        if (LeftHand.GetFingerIsPinching(OVRHand.HandFinger.Pinky))
        {
            StreamRelativeData = false;
            StreamAbsoluteRightData = true;
            StreamAbsoluteLeftData = true;
            StreamResolution = true;
            PauseLeft= true;
            PauseRight= true;
            StreamBorder.color = Color.yellow; // Black color for resolution
            ToggleMenuButton(false);
            ToggleResolutionButton(false);
            WristTracker.SetActive(true);

        }

        if (RightHand.GetFingerIsPinching(OVRHand.HandFinger.Pinky))
        {
            StreamRelativeData = false;
            StreamAbsoluteRightData = true;
            StreamAbsoluteLeftData = true;
            StreamResolution = false;
            StreamBorder.color = Color.yellow; // Black color for resolution
            PauseLeft= true;
            PauseRight= true;
            ToggleMenuButton(false);
            ToggleResolutionButton(false);
            WristTracker.SetActive(true);
        }    
    }

    void Update()
    {
        if (rightconnectionEstablished && leftconnectionEstablished)
        {   
            SendResolution();
            if (String.Equals(rightcommunicationAddress, netConfig.getRightKeypointAddress())||String.Equals(leftcommunicationAddress, netConfig.getLeftKeypointAddress()))
            {    
                StreamPauser();
                if(StreamAbsoluteLeftData && StreamAbsoluteRightData)
                {
                    SendRightHandData("absolute");
                    SendLeftHandData("absolute");
                    byte[] recievedleftToken = leftclient.ReceiveFrameBytes();
                    byte[] recievedrightToken = rightclient.ReceiveFrameBytes();
                    ToggleResolutionButton(false);
                }
                if (StreamRelativeData)
                {
                    SendRightHandData("relative");
                    SendLeftHandData("relative");
                    byte[] recievedrightToken = rightclient.ReceiveFrameBytes();
                    byte[] recievedleftToken = leftclient.ReceiveFrameBytes();
                    ToggleResolutionButton(false);    
                }

                if (StreamResolution)
                {   
                    ToggleHighResolutionButton(true);
                    ToggleLowResolutionButton(true);
                }  
              
            }
            else
            {
                connectionEstablished = false;
            }
        
        } else
        {
            StreamBorder.color = Color.red;
            ToggleMenuButton(true);
            CreateTCPConnection();     
        }
    }
}
