/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using System.Collections.Generic;
using UnityEngine;

namespace Oculus.Interaction.Input
{
    /// <summary>
    /// Returns the active status of an OVRInput device based on whether
    /// OVRInput's current active controller matches any of the controller
    /// types set up in the inspector. OVRInput `Controllers` include
    /// types like Touch, L Touch, R TouchR, Hands, L Hand, R Hand
    /// </summary>
    public class OVRInputDeviceActiveState : MonoBehaviour, IActiveState
    {
        [SerializeField]
        private List<OVRInput.Controller> _controllerTypes;

        public bool Active
        {
            get
            {
                foreach (OVRInput.Controller controllerType in _controllerTypes)
                {
                    if (OVRInput.GetConnectedControllers() == controllerType) return true;
                }
                return false;
            }
        }

        #region Inject

        public void InjectAllOVRInputDeviceActiveState(List<OVRInput.Controller> controllerTypes)
        {
            InjectControllerTypes(controllerTypes);
        }

        public void InjectControllerTypes(List<OVRInput.Controller> controllerTypes)
        {
            _controllerTypes = controllerTypes;
        }

        #endregion
    }
}
