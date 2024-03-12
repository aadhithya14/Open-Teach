/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

namespace Oculus.Interaction
{
    /// <summary>
    /// Represents a curved rectangular section of a
    /// cylinder wall.
    /// </summary>
    public interface ICurvedPlane
    {
        /// <summary>
        /// The cylinder the curved plane lies on
        /// </summary>
        Cylinder Cylinder { get; }

        /// <summary>
        /// The horizontal size of the plane, in degrees
        /// </summary>
        float ArcDegrees { get; }

        /// <summary>
        /// The rotation of the center of the plane relative
        /// to the Cylinder's forward Z axis, in degrees
        /// </summary>
        float Rotation { get; }

        /// <summary>
        /// The bottom of the plane relative to the
        /// Cylinder Y position, in Cylinder local space
        /// </summary>
        float Bottom { get; }

        /// <summary>
        /// The top of the plane relative to the
        /// Cylinder Y position, in Cylinder local space
        /// </summary>
        float Top { get; }
    }
}
