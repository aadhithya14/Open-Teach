/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using UnityEngine;
namespace Oculus.Interaction
{
    [DefaultExecutionOrder(-1)]
    public class InteractionOVRCameraRig : OVRCameraRig
    {
        private bool _isLateUpdate = false;

        public event System.Action<bool> WhenInputDataDirtied = delegate { };

        protected override void OnBeforeRenderCallback()
        {
            _isLateUpdate = true;
            base.OnBeforeRenderCallback();
            _isLateUpdate = false;
        }

        protected override void RaiseUpdatedAnchorsEvent()
        {
            base.RaiseUpdatedAnchorsEvent();
            WhenInputDataDirtied(_isLateUpdate);
        }
    }
}
