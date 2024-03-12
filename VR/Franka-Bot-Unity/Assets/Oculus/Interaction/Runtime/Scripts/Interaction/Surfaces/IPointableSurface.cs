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

namespace Oculus.Interaction.Surfaces
{
    public interface IPointableSurface
    {
        /// <summary>
        /// Find the distance between two world space surface points
        /// in the surface's frame of reference.
        /// </summary>
        /// <param name="worldPoint0">World point on surface</param>
        /// <param name="worldPoint1">World point on surface</param>
        /// <returns>Vector2(Horizontal delta, Vertical delta)</returns>
        Vector2 GetSurfaceDistanceBetween(Vector3 worldPoint0, Vector3 worldPoint1);

        /// <summary>
        /// Check if a point is on the normal-side of the collider
        /// (is raycast hit possible from this origin point)
        /// </summary>
        /// <param name="point">The point to check</param>
        /// <returns>True when above the surface</returns>
        bool IsPointAboveSurface(Vector3 point);

        /// <summary>
        /// Raycast to the surface with an optional maximum distance value
        /// </summary>
        /// <param name="ray">The ray to cast</param>
        /// <param name="hit">The returned hit data</param>
        /// <param name="maxDistance">If greater than zero, maximum distance of raycast</param>
        /// <returns>true if surface was hit</returns>
        bool Raycast(in Ray ray, out SurfaceHit hit, float maxDistance = 0);

        /// <summary>
        /// Find nearest point to surface
        /// </summary>
        /// <param name="ray">Point to check</param>
        /// <param name="hit">The returned hit data</param>
        /// <param name="maxDistance">If greater than zero, maximum distance of check</param>
        /// <returns>true if nearest point was found</returns>
        bool ClosestSurfacePoint(in Vector3 point, out SurfaceHit hit, float maxDistance = 0);
    }

    public struct SurfaceHit
    {
        /// <summary>
        /// The hit point on the surface
        /// </summary>
        public Vector3 Point { get; set; }

        /// <summary>
        /// The normal of the surface hit
        /// </summary>
        public Vector3 Normal { get; set; }

        /// <summary>
        /// The distance of the hit from the origin point
        /// </summary>
        public float Distance { get; set; }
    }
}
