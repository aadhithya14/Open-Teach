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

namespace Oculus.Interaction.Samples
{
    public class StayInView : MonoBehaviour
    {
        [SerializeField]
        private Transform _eyeCenter;

        [SerializeField]
        private float _extraDistanceForward = 0;

        [SerializeField]
        private bool _zeroOutEyeHeight = true;
        void Update()
        {
            transform.rotation = Quaternion.identity;
            transform.position = _eyeCenter.position;
            transform.Rotate(0, _eyeCenter.rotation.eulerAngles.y, 0, Space.Self);
            transform.position = _eyeCenter.position + transform.forward.normalized * _extraDistanceForward;
            if (_zeroOutEyeHeight)
                transform.position = new Vector3(transform.position.x, 0, transform.position.z);
        }
    }
}
