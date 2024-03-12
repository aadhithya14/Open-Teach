/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using System;
using UnityEngine;

namespace Oculus.Interaction.Input
{
    /// <summary>
    /// Describes where the pose data originated. Can be used to determine how much pre-processing
    /// has been applied by modifiers. This can be useful in determining how to render the hands.
    /// </summary>
    public enum PoseOrigin
    {
        /// <summary>
        /// Pose is invalid and has no meaning.
        /// </summary>
        None,

        /// <summary>
        /// Pose matches this frames tracking data; no filtering or modification has occured.
        /// </summary>
        RawTrackedPose,

        /// <summary>
        /// Pose originated from this frames tracking data but has had additional filtering or
        /// modification applied by an IInputDataModifier
        /// </summary>
        FilteredTrackedPose,

        /// <summary>
        /// Pose is valid but was not derived from this frames tracking data. Examples include
        /// last-known-good data when tracking is lost, or puppet-hands for tutorials.
        /// </summary>
        SyntheticPose,
    }
}
