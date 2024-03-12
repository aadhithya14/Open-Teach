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
    public static partial class Collisions
    {
        public static Vector3 ClosestPointToColliders(Vector3 point, Collider[] colliders)
        {
            Vector3 closestPoint = point;
            float closestDistance = float.MaxValue;
            foreach (Collider collider in colliders)
            {
                if (Collisions.IsPointWithinCollider(point, collider))
                {
                    return point;
                }

                Vector3 closest = collider.ClosestPoint(point);
                float distance = (closest - point).magnitude;
                if (distance < closestDistance)
                {
                    closestDistance = distance;
                    closestPoint = closest;
                }
            }

            return closestPoint;
        }
    }
}
