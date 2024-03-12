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
using UnityEngine.Assertions;
using UnityEngine.Serialization;

namespace Oculus.Interaction
{
    public class HandJoint : MonoBehaviour
    {
        [SerializeField, Interface(typeof(IHand))]
        private MonoBehaviour _hand;
        public IHand Hand { get; private set; }

        [SerializeField]
        private HandJointId _handJointId;

        [SerializeField]
        private Vector3 _localPositionOffset;

        [SerializeField]
        private Quaternion _rotationOffset = Quaternion.identity;

        #region Properties

        public HandJointId HandJointId
        {
            get
            {
                return _handJointId;
            }
            set
            {
                _handJointId = value;
            }
        }

        public Vector3 LocalPositionOffset
        {
            get
            {
                return _localPositionOffset;
            }
            set
            {
                _localPositionOffset = value;
            }
        }

        public Quaternion RotationOffset
        {
            get
            {
                return _rotationOffset;
            }
            set
            {
                _rotationOffset = value;
            }
        }

        #endregion

        protected bool _started = false;

        protected virtual void Awake()
        {
            Hand = _hand as IHand;
        }

        protected virtual void Start()
        {
            this.BeginStart(ref _started);
            Assert.IsNotNull(Hand);
            this.EndStart(ref _started);
        }

        protected virtual void OnEnable()
        {
            if (_started)
            {
                Hand.WhenHandUpdated += HandleHandUpdated;
            }
        }

        protected virtual void OnDisable()
        {
            if (_started)
            {
                Hand.WhenHandUpdated -= HandleHandUpdated;
            }
        }

        private void HandleHandUpdated()
        {
            if (!Hand.GetJointPose(_handJointId, out Pose pose)) return;

            Vector3 positionOffsetWithHandedness =
                (Hand.Handedness == Handedness.Left ? -1f : 1f) * _localPositionOffset;
            pose.position += _rotationOffset * pose.rotation *
                              positionOffsetWithHandedness * Hand.Scale;
            transform.SetPose(pose);
        }

        #region Inject

        public void InjectAllHandJoint(IHand hand)
        {
            InjectHand(hand);
        }

        public void InjectHand(IHand hand)
        {
            _hand = hand as MonoBehaviour;
            Hand = hand;
        }

        #endregion;
    }
}
