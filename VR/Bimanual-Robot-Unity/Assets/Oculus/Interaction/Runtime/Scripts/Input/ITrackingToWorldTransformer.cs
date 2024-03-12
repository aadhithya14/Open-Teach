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

namespace Oculus.Interaction.Input
{
    public interface ITrackingToWorldTransformer
    {
        Transform Transform { get; }

        /// <summary>
        /// Converts a tracking space pose to a pose in in Unity's world coordinate space
        /// (i.e. teleportation applied)
        /// </summary>
        Pose ToWorldPose(Pose poseRh);

        /// <summary>
        /// Converts a world space pose in Unity's coordinate space
        /// to a pose in tracking space (i.e. no teleportation applied)
        /// </summary>
        Pose ToTrackingPose(in Pose worldPose);

        Quaternion WorldToTrackingWristJointFixup { get; }
    }
}
