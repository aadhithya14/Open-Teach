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
    public class CircleProximityField : MonoBehaviour, IProximityField
    {
        [SerializeField]
        private Transform _transform;

        [SerializeField]
        private float _radius = 0.1f;

        protected virtual void Start()
        {
            Assert.IsNotNull(_transform);
        }

        // Closest point to circle is computed by projecting point to the plane
        // the circle is on and then clamping to the circle
        public Vector3 ComputeClosestPoint(Vector3 point)
        {
            Vector3 vectorFromPlane = point - _transform.position;
            Vector3 planeNormal = -1.0f * _transform.forward;
            Vector3 projectedPoint = Vector3.ProjectOnPlane(vectorFromPlane, planeNormal);

            float distanceFromCenterSqr = projectedPoint.sqrMagnitude;
            float worldRadius = transform.lossyScale.x * _radius;
            if (distanceFromCenterSqr > worldRadius * worldRadius)
            {
                projectedPoint = worldRadius * projectedPoint.normalized;
            }
            return projectedPoint + _transform.position;
        }

        #region Inject
        public void InjectAllCircleProximityField(Transform centerTransform)
        {
            InjectCenterTransform(centerTransform);
        }

        public void InjectCenterTransform(Transform centerTransform)
        {
            _transform = centerTransform;
        }

        public void InjectOptionalRadius(float radius)
        {
            _radius = radius;
        }

        #endregion
    }
}
