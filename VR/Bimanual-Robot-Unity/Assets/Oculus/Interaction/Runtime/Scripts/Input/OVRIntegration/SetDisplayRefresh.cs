/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using System.Linq;
using UnityEngine;

namespace Oculus.Interaction.Input
{
    public class SetDisplayRefresh : MonoBehaviour
    {
        [SerializeField]
        private float _desiredDisplayFrequency = 90f;

        public void SetDesiredDisplayFrequency(float desiredDisplayFrequency)
        {
            var validFrequencies = OVRPlugin.systemDisplayFrequenciesAvailable;

            if (validFrequencies.Contains(_desiredDisplayFrequency))
            {
                Debug.Log("[Oculus.Interaction] Setting desired display frequency to " + _desiredDisplayFrequency);
                OVRPlugin.systemDisplayFrequency = _desiredDisplayFrequency;
            }
        }

        protected virtual void Awake()
        {
            SetDesiredDisplayFrequency(_desiredDisplayFrequency);
        }
    }
}
