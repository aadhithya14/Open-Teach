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

namespace Oculus.Interaction
{
    public class PointProximityField : MonoBehaviour, IProximityField
    {
        [SerializeField]
        private Transform _centerPoint;

        protected virtual void Start()
        {
            Assert.IsNotNull(_centerPoint);
        }

        public Vector3 ComputeClosestPoint(Vector3 point)
        {
            return _centerPoint.position;
        }

        #region Inject

        public void InjectAllPointProximityField(Transform centerPoint)
        {
            InjectCenterPoint(centerPoint);
        }

        public void InjectCenterPoint(Transform centerPoint)
        {
            _centerPoint = centerPoint;
        }

        #endregion
    }
}
