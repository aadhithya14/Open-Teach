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
using UnityEngine.Assertions;

namespace Oculus.Interaction.HandGrab.Visuals
{
    /// <summary>
    /// A static (non-user controlled) representation of a hand. This script is used
    /// to be able to manually visualize hand grab poses.
    /// </summary>
    [RequireComponent(typeof(HandPuppet))]
    public class HandGhost : MonoBehaviour
    {
        /// <summary>
        /// The puppet is used to actually move the representation of the hand.
        /// </summary>
        [SerializeField]
        private HandPuppet _puppet;

        /// <summary>
        /// The HandGrab point can be set so the ghost automatically
        /// adopts the desired pose of said point.
        /// </summary>
        [SerializeField, Optional]
        [UnityEngine.Serialization.FormerlySerializedAs("_handGrabPoint")]
        private HandGrabPose _handGrabPose;

        #region editor events
        protected virtual void Reset()
        {
            _puppet = this.GetComponent<HandPuppet>();
            _handGrabPose = this.GetComponentInParent<HandGrabPose>();
        }

        protected virtual void OnValidate()
        {
            if (_puppet == null)
            {
                return;
            }

            if (_handGrabPose == null)
            {
                HandGrabPose point = this.GetComponentInParent<HandGrabPose>();
                if (point != null)
                {
                    SetPose(point);
                }
            }
            else if (_handGrabPose != null)
            {
                SetPose(_handGrabPose);
            }
        }
        #endregion

        protected virtual void Start()
        {
            Assert.IsNotNull(_puppet);
        }

        /// <summary>
        /// Relay to the Puppet to set the ghost hand to the desired static pose
        /// </summary>
        /// <param name="handGrabPose">The point to read the HandPose from</param>
        public void SetPose(HandGrabPose handGrabPose)
        {
            HandPose userPose = handGrabPose.HandPose;
            if (userPose == null)
            {
                return;
            }

            Transform relativeTo = handGrabPose.RelativeTo;
            _puppet.SetJointRotations(userPose.JointRotations);
            SetRootPose(handGrabPose.RelativeGrip, relativeTo);
        }

        /// <summary>
        /// Moves the underlying puppet so the wrist point aligns with the given parameters
        /// </summary>
        /// <param name="rootPose">The relative wrist pose to align the hand to</param>
        /// <param name="relativeTo">The object to use as anchor</param>
        public void SetRootPose(Pose rootPose, Transform relativeTo)
        {
            rootPose.Postmultiply(relativeTo.GetPose());
            _puppet.SetRootPose(rootPose);
        }

        #region Inject
        public void InjectAllHandGhost(HandPuppet puppet, Transform gripPoint)
        {
            InjectHandPuppet(puppet);
        }
        public void InjectHandPuppet(HandPuppet puppet)
        {
            _puppet = puppet;
        }
        public void InjectOptionalHandGrabPose(HandGrabPose handGrabPose)
        {
            _handGrabPose = handGrabPose;
        }

        #endregion
    }
}
