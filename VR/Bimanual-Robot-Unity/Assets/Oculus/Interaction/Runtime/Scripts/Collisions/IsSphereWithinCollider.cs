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
        public static bool IsSphereWithinCollider(Vector3 point, float radius, Collider collider)
        {
            Vector3 closestPoint = collider.ClosestPoint(point);
            if (closestPoint.Equals(point)) return true;
            if (Vector3.SqrMagnitude(closestPoint - point) < radius * radius) return true;
            return false;
        }
    }
}
