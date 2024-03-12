using UnityEngine;
using UnityEngine.UI;


public class HighResolutionButtonController : MonoBehaviour
{
    // This variable will be set to true when the button is clicked
    public bool HighResolution = false;
    // public LowResolutionButtonController LowResolutionButtonController;

    // Attach this method to the button's OnClick event in the Unity Editor
    public void OnButtonClick()
    {
        HighResolution = true;
        // LowResolutionButtonController.LowResolution=false;
    }
    
}
