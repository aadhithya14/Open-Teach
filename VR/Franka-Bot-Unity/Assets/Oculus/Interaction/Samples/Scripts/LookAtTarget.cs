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
using UnityEngine.Assertions;

namespace Oculus.Interaction.Samples
{
    public class LookAtTarget : MonoBehaviour
    {
        [SerializeField]
        private Transform _toRotate;

        [SerializeField]
        private Transform _target;

        protected virtual void Start()
        {
            Assert.IsNotNull(_toRotate);
            Assert.IsNotNull(_target);
        }

        protected virtual void Update()
        {
            Vector3 dirToTarget = (_target.position - _toRotate.position).normalized;
            _toRotate.LookAt(_toRotate.position - dirToTarget, Vector3.up);
        }
    }
}
