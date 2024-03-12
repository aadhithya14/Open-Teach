/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using Oculus.Interaction.Input;
using UnityEngine;

namespace Oculus.Interaction.HandGrab
{
    /// <summary>
    /// Defines the strategy for aligning the hand to the snapped object.
    /// The hand can go to the object upon selection, during hover or
    /// simply stay in its pose.
    /// </summary>
    public enum HandAlignType
    {
        AlignOnGrab,
        AttractOnHover,
        None
    }

    /// <summary>
    /// Interface for interactors that allow aligning to an object.
    /// Contains information to drive the HandGrabVisual moving
    /// the fingers and wrist.
    /// </summary>
    public interface IHandGrabState
    {
        bool IsGrabbing { get; }
        float GrabStrength { get; }
        Pose WristToGrabPoseOffset { get; }
        HandFingerFlags GrabbingFingers();
        HandGrabTarget HandGrabTarget { get; }
        System.Action<IHandGrabState> WhenHandGrabStarted { get; set; }
        System.Action<IHandGrabState> WhenHandGrabEnded { get; set; }
    }
}
