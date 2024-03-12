using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;

public class HandPointerLike : MonoBehaviour
{
    public OVRInputModule _OVRInputModule;
    public OVRRaycaster _OVRRaycaster;
    public OVRHand _OVRHand;

    // Start is called before the first frame update
    void Start()
    {
        _OVRInputModule.rayTransform = _OVRHand.PointerPose;
        _OVRRaycaster.pointer = _OVRHand.PointerPose.gameObject;
    }

    // Update is called once per frame
    void Update()
    {
        _OVRInputModule.rayTransform = _OVRHand.PointerPose;
        _OVRRaycaster.pointer = _OVRHand.PointerPose.gameObject;
    }
}
