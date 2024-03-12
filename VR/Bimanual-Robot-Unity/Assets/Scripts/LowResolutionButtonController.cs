using UnityEngine;
using UnityEngine.UI;

public class LowResolutionButtonController : MonoBehaviour
{
    // This variable will be set to true when the button is clicked
    public bool LowResolution = false;
    //public HighResolutionButtonController HighResolutionButtonController;

    // Attach this method to the button's OnClick event in the Unity Editor
    public void OnButtonClick()
    {
        LowResolution = true;
        //HighResolutionButtonController.HighResolution=true;
    }
}
