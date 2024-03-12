using UnityEngine;
using UnityEngine.UI;

using NetMQ;
using NetMQ.Sockets;

using System;
using System.Collections.Generic;
using System.Threading;

public class GraphStream : MonoBehaviour
{
    private Thread graphStreamer;
    private static List<byte[]> graphList;

    public RawImage image;
    private Texture2D texture;

    private bool connectionEstablished = false;
    private string communicationAddress;
    private NetworkManager netConfig;
    private SubscriberSocket socket;

    private void StartGraphThread()
    {
        // Check if communication address is available
        communicationAddress = netConfig.getGraphAddress();
        bool AddressAvailable = !String.Equals(communicationAddress, "tcp://:");
        
        if (AddressAvailable)
        {
            StartConnection();
            graphList = new List<byte[]>();
            graphStreamer = new Thread(getGraphImage);
            graphStreamer.Start();
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

    private void getGraphImage()
    {
        while (true)
        {
            byte[] imageBytes = socket.ReceiveFrameBytes();
            graphList.Add(imageBytes);

            if (graphList.Count > 2)
            {
                graphList.RemoveAt(0);
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
            if (String.Equals(communicationAddress, netConfig.getGraphAddress()))
            {
                // Getting the image from the queue and displaying it
                byte[] imageBytes = graphList[graphList.Count - 1];
                texture.LoadImage(imageBytes);
            }
            else
            {
                // Aborting the queue
                graphStreamer.Abort();
                connectionEstablished = false;
            }
        }
        else
        {
            StartGraphThread();
        }
    }
}
