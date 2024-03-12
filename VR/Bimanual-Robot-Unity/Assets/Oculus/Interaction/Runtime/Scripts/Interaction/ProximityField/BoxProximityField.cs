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
    public class BoxProximityField : MonoBehaviour, IProximityField
    {
        [SerializeField]
        private Transform _boxTransform;

        protected virtual void Start()
        {
            Assert.IsNotNull(_boxTransform);
        }

        // Closest point in box is computed by transforming the point to OBB space,
        // clamping to a 1-1-1 box, and transforming the point back to world space
        public Vector3 ComputeClosestPoint(Vector3 point)
        {
            Vector3 localPoint = _boxTransform.InverseTransformPoint(point);

            localPoint.x = Mathf.Clamp(localPoint.x, -0.5f, 0.5f);
            localPoint.y = Mathf.Clamp(localPoint.y, -0.5f, 0.5f);
            localPoint.z = Mathf.Clamp(localPoint.z, -0.5f, 0.5f);

            Vector3 worldPoint = _boxTransform.TransformPoint(localPoint);

            return worldPoint;
        }

        #region Inject

        public void InjectAllBoxProximityField(Transform boxTransform)
        {
            InjectBoxTransform(boxTransform);
        }

        public void InjectBoxTransform(Transform boxTransform)
        {
            _boxTransform = boxTransform;
        }

        #endregion

    }
}
