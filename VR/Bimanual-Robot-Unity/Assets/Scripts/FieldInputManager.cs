using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class FieldInputManager : MonoBehaviour
{
    public TMP_Dropdown FirstDropDown;
    public TMP_Dropdown SecondDropDown;
    public TMP_Dropdown ThirdDropDown;
    public TMP_Dropdown FourthDropDown;

    private NetworkManager netConfig;

    void Start()
    {
        List<string> optionList = new List<string>();

        // Create a for loop which generates all the options
        for (int optionNumber = 0; optionNumber < 256; optionNumber++)
            optionList.Add(optionNumber.ToString());

        FirstDropDown.AddOptions(optionList);
        SecondDropDown.AddOptions(optionList);
        ThirdDropDown.AddOptions(optionList);
        FourthDropDown.AddOptions(optionList);

        // Getting the Network Config Updater gameobject
        GameObject netConfGame = GameObject.Find("NetworkConfigsLoader");
        netConfig = netConfGame.GetComponent<NetworkManager>();
    }

    public void getIPAddress()
    {
        string newIPAddress = FirstDropDown.options[FirstDropDown.value].text + "." +
            SecondDropDown.options[SecondDropDown.value].text + "." +
            ThirdDropDown.options[ThirdDropDown.value].text + "." +
            FourthDropDown.options[FourthDropDown.value].text;

        // Change the IP Address
        netConfig.changeIPAddress(newIPAddress);
    }
}
