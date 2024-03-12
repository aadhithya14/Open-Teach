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

namespace Oculus.Interaction
{
    public class CenterEyeOffset : MonoBehaviour
    {
        [SerializeField, Interface(typeof(IHand))]
        private MonoBehaviour _hand;
        public IHand Hand { get; private set; }

        [SerializeField]
        private Vector3 _offset;

        [SerializeField]
        private Quaternion _rotation = Quaternion.identity;

        private Pose _cachedPose = Pose.identity;

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
            if (Hand.GetCenterEyePose(out Pose rootPose))
            {
                GetOffset(ref _cachedPose);
                _cachedPose.Postmultiply(rootPose);
                transform.SetPose(_cachedPose);
            }
        }

        public void GetOffset(ref Pose pose)
        {
            pose.position = _offset;
            pose.rotation = _rotation;
        }

        public void GetWorldPose(ref Pose pose)
        {
            pose.position = this.transform.position;
            pose.rotation = this.transform.rotation;
        }

        #region Inject
        public void InjectHand(IHand hand)
        {
            _hand = hand as MonoBehaviour;
            Hand = hand;
        }

        public void InjectOffset(Vector3 offset)
        {
            _offset = offset;
        }

        public void InjectRotation(Quaternion rotation)
        {
            _rotation = rotation;
        }

        public void InjectAllCenterEyeOffset(IHand hand,
            Vector3 offset, Quaternion rotation)
        {
            InjectHand(hand);
            InjectOffset(offset);
            InjectRotation(rotation);
        }
        #endregion
    }
}
