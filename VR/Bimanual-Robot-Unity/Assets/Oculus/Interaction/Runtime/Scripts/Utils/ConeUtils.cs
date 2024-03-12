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
    public class ConeUtils
    {
        public static bool RayWithinCone(Ray ray, Vector3 position, float apertureDegrees)
        {
            float minDotProductThreshold = Mathf.Cos(apertureDegrees * Mathf.Deg2Rad);

            var vectorToInteractable = position - ray.origin;
            var distanceToInteractable = vectorToInteractable.magnitude;

            if (Mathf.Abs(distanceToInteractable) < 0.001f) return true;

            vectorToInteractable /= distanceToInteractable;
            var dotProduct = Vector3.Dot(vectorToInteractable, ray.direction);

            return dotProduct >= minDotProductThreshold;
        }
    }
}
