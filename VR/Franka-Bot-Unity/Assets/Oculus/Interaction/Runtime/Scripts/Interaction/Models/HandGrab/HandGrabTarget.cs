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

namespace Oculus.Interaction.HandGrab
{
    /// <summary>
    /// All the relevant data needed for a snapping position.
    /// This includes the Interactor and the surface (if any) around
    /// which the pose is valid.
    /// </summary>
    public class HandGrabTarget
    {
        public enum GrabAnchor
        {
            None,
            Wrist,
            Pinch,
            Palm
        }

        public ref Pose Pose => ref _pose;

        public HandPose HandPose => _isHandPoseValid ? _handPose : null;

        public Pose WorldGrabPose => _relativeTo != null ? _relativeTo.GlobalPose(Pose) : Pose.identity;
        public HandAlignType HandAlignment => _handAlignment;

        public GrabAnchor Anchor { get; private set; } = GrabAnchor.None;

        private bool _isHandPoseValid = false;
        private HandPose _handPose = new HandPose();
        private Pose _pose;

        private Transform _relativeTo;
        private HandAlignType _handAlignment;


        public void Set(HandGrabTarget other)
        {
            Set(other._relativeTo, other._handAlignment, other.HandPose, other._pose, other.Anchor);
        }

        public void Set(Transform relativeTo, HandAlignType handAlignment, HandPose pose, in Pose snapPoint, GrabAnchor anchor)
        {
            Anchor = anchor;
            _relativeTo = relativeTo;
            _handAlignment = handAlignment;
            _pose.CopyFrom(snapPoint);
            _isHandPoseValid = pose != null;
            if (_isHandPoseValid)
            {
                _handPose.CopyFrom(pose);
            }
        }

        public void Clear()
        {
            Anchor = GrabAnchor.None;
            _isHandPoseValid = false;
            _relativeTo = null;
            _handAlignment = HandAlignType.None;
        }
    }
}
