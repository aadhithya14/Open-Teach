/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using Oculus.Interaction.HandGrab.SnapSurfaces;
using Oculus.Interaction.Input;
using UnityEngine;
using UnityEngine.Assertions;

namespace Oculus.Interaction.HandGrab
{
    [System.Serializable]
    public struct HandGrabPoseData
    {
        public Pose gripPose;
        public HandPose handPose;
        public float scale;
    }

    /// <summary>
    /// The HandGrabPose defines the local point in an object to which the grip point
    /// of the hand should align. It can also contain information about the final pose
    /// of the hand for perfect alignment as well as a surface that indicates the valid
    /// positions for the point.
    /// </summary>
    public class HandGrabPose : MonoBehaviour
    {
        [SerializeField]
        private Transform _relativeTo;

        [SerializeField, Optional, Interface(typeof(ISnapSurface))]
        private MonoBehaviour _surface = null;
        private ISnapSurface _snapSurface;
        public ISnapSurface SnapSurface
        {
            get => _snapSurface ?? _surface as ISnapSurface;
            private set
            {
                _snapSurface = value;
            }
        }

        [SerializeField]
        private bool _usesHandPose = true;

        [SerializeField, Optional]
        [HideInInspector]
        private HandPose _handPose = new HandPose();

        public HandPose HandPose => _usesHandPose ? _handPose : null;
        public float Scale => this.transform.lossyScale.x;
        public Transform RelativeTo { get => _relativeTo; set => _relativeTo = value; }
        public Pose RelativeGrip => RelativeTo.RelativeOffset(this.transform);

        #region editor events

        protected virtual void Reset()
        {
            _relativeTo = this.GetComponentInParent<HandGrabInteractable>()?.RelativeTo;
        }
        #endregion

        protected virtual void Start()
        {
            Assert.IsNotNull(_relativeTo, "The relative transform can not be null");
        }

        /// <summary>
        /// Applies the given position/rotation to the HandGrabPose
        /// </summary>
        /// <param name="handPose">Relative hand position/rotation.</param>
        /// <param name="relativeTo">Reference coordinates for the pose.</param>
        public void SetPose(HandPose handPose, in Pose gripPoint, Transform relativeTo)
        {
            _handPose = new HandPose(handPose);
            _relativeTo = relativeTo;
            this.transform.SetPose(relativeTo.GlobalPose(gripPoint));
        }

        public HandGrabPoseData SaveData()
        {
            HandGrabPoseData data = new HandGrabPoseData()
            {
                handPose = new HandPose(_handPose),
                scale = Scale,
                gripPose = _relativeTo.RelativeOffset(this.transform)
            };

            return data;
        }

        public void LoadData(HandGrabPoseData data, Transform relativeTo)
        {
            _relativeTo = relativeTo;
            this.transform.localScale = Vector3.one * data.scale;
            this.transform.SetPose(_relativeTo.GlobalPose(data.gripPose));
            if (data.handPose != null)
            {
                _handPose = new HandPose(data.handPose);
            }
        }

        public virtual bool CalculateBestPose(Pose userPose, Handedness handedness,
            ref HandPose bestHandPose, ref Pose bestSnapPoint, in PoseMeasureParameters scoringModifier,
            out bool usesHandPose, out float score)
        {
            usesHandPose = false;
            if (HandPose != null && HandPose.Handedness != handedness)
            {
                score = float.NaN;
                return false;
            }

            score = CompareNearPoses(userPose, ref bestSnapPoint, scoringModifier);
            if (HandPose != null)
            {
                usesHandPose = true;
                bestHandPose.CopyFrom(HandPose);
            }

            return true;
        }

        /// <summary>
        /// Finds the most similar pose at this HandGrabInteractable to the user hand pose
        /// </summary>
        /// <param name="worldPoint">The user current hand pose.</param>
        /// <param name="bestSnapPoint">The snap point hand pose within the surface (if any).</param>
        /// <returns>The adjusted best pose at the surface.</returns>
        private float CompareNearPoses(in Pose worldPoint, ref Pose bestSnapPoint, in PoseMeasureParameters scoringModifier)
        {
            Pose desired = worldPoint;
            Pose snap = this.transform.GetPose();

            float bestScore;
            Pose bestPlace;
            if (SnapSurface != null)
            {
                bestScore = SnapSurface.CalculateBestPoseAtSurface(desired, snap, out bestPlace, scoringModifier);
            }
            else
            {
                bestPlace = snap;
                bestScore = PoseUtils.Similarity(desired, snap, scoringModifier);
            }

            _relativeTo.RelativeOffset(bestPlace, ref bestSnapPoint);

            return bestScore;
        }

        #region Inject

        public void InjectRelativeTo(Transform relativeTo)
        {
            _relativeTo = relativeTo;
        }

        public void InjectOptionalSurface(ISnapSurface surface)
        {
            _surface = surface as MonoBehaviour;
            SnapSurface = surface;
        }

        public void InjectOptionalHandPose(HandPose handPose)
        {
            _handPose = handPose;
            _usesHandPose = _handPose != null;
        }

        public void InjectAllHandGrabPose(Transform relativeTo)
        {
            InjectRelativeTo(relativeTo);
        }
        #endregion

    }
}
