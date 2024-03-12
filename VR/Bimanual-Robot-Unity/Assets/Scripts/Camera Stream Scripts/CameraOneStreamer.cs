using UnityEngine;
using UnityEngine.UI;

using NetMQ;
using NetMQ.Sockets;

using System;
using System.Collections.Generic;
using System.Threading;

public class CameraOneStreamer : MonoBehaviour
{
    private Thread imageStreamer;
    private static List<byte[]> imageList;

    public RawImage image;
    private Texture2D texture;

    //public NetworkConfigs netConf;
    private bool connectionEstablished = false;
    private string communicationAddress;
    private NetworkManager netConfig;
    private SubscriberSocket socket;

    private void StartImageThread()
    {
        // Check if communication address is available
        communicationAddress = netConfig.getCamAddress();
        bool AddressAvailable = !String.Equals(communicationAddress, "tcp://:");

        if (AddressAvailable)
        {
            StartConnection();
            imageList = new List<byte[]>();
            imageStreamer = new Thread(getRobotImage);
            imageStreamer.Start();
        }
    }

    public void StartConnection()
    {
        // Initiate Subscriber Socket
        socket = new SubscriberSocket();
        socket.Options.ReceiveHighWatermark = 1000;
        socket.Connect(communicationAddress);
        socket.Subscribe("");
        connectionEstablished = true;
    }

    private void getRobotImage()
    {
        while (true)
        {
            byte[] imageBytes = socket.ReceiveFrameBytes();
            imageList.Add(imageBytes);

            if (imageList.Count > 5)
            {
                imageList.RemoveAt(0);
            }
        }
    }

    public void Start()
    {
        // Getting the Network Config Updater gameobject
        GameObject netConfGame = GameObject.Find("NetworkConfigsLoader");
        netConfig = netConfGame.GetComponent<NetworkManager>();

        // Initializing the image texture
        texture = new Texture2D(640, 360, TextureFormat.RGB24, false);
        image.texture = texture;
    }

    public void Update()
    {
        if (connectionEstablished)
        {
            // To check if the same IP is being used
            if (String.Equals(communicationAddress, netConfig.getCamAddress()))
            {
                // Getting the image from the queue and displaying it
                byte[] imageBytes = imageList[imageList.Count - 1];
                texture.LoadImage(imageBytes);
            }
            else
            {
                // Aborting the queue
                imageStreamer.Abort();
                connectionEstablished = false;
            }
        } else
        {
            StartImageThread();
        }
    }
}