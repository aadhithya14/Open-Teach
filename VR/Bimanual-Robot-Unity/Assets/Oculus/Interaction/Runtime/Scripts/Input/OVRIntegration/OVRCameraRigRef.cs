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
using UnityEngine.Assertions;

namespace Oculus.Interaction.Input
{
    public interface IOVRCameraRigRef
    {
        OVRCameraRig CameraRig { get; }
        /// <summary>
        /// Returns a valid OVRHand object representing the left hand, if one exists on the
        /// OVRCameraRig. If none is available, returns null.
        /// </summary>
        OVRHand LeftHand { get; }
        /// <summary>
        /// Returns a valid OVRHand object representing the right hand, if one exists on the
        /// OVRCameraRig. If none is available, returns null.
        /// </summary>
        OVRHand RightHand { get; }
        Transform LeftController { get; }
        Transform RightController { get; }

        event Action<bool> WhenInputDataDirtied;
    }

    /// <summary>
    /// Points to an OVRCameraRig instance. This level of indirection provides a single
    /// configuration point on the root of a prefab.
    /// Must execute before all other OVR related classes so that the fields are
    /// initialized correctly and ready to use.
    /// </summary>
    [DefaultExecutionOrder(-90)]
    public class OVRCameraRigRef : MonoBehaviour, IOVRCameraRigRef
    {
        [Header("Configuration")]
        [SerializeField]
        private InteractionOVRCameraRig _ovrCameraRig;

        [SerializeField]
        private bool _requireOvrHands = true;

        public OVRCameraRig CameraRig => _ovrCameraRig;

        private OVRHand _leftHand;
        private OVRHand _rightHand;
        public OVRHand LeftHand => GetHandCached(ref _leftHand, _ovrCameraRig.leftHandAnchor);
        public OVRHand RightHand => GetHandCached(ref _rightHand, _ovrCameraRig.rightHandAnchor);

        public Transform LeftController => _ovrCameraRig.leftControllerAnchor;
        public Transform RightController => _ovrCameraRig.rightControllerAnchor;

        public event Action<bool> WhenInputDataDirtied = delegate { };

        protected bool _started = false;

        protected virtual void Start()
        {
            this.BeginStart(ref _started);
            Assert.IsNotNull(_ovrCameraRig);
            this.EndStart(ref _started);
        }

        protected virtual void OnEnable()
        {
            if (_started)
            {
                _ovrCameraRig.WhenInputDataDirtied += HandleInputDataDirtied;
            }
        }

        protected virtual void OnDisable()
        {
            if (_started)
            {
                _ovrCameraRig.WhenInputDataDirtied -= HandleInputDataDirtied;
            }
        }

        private OVRHand GetHandCached(ref OVRHand cachedValue, Transform handAnchor)
        {
            if (cachedValue != null)
            {
                return cachedValue;
            }

            cachedValue = handAnchor.GetComponentInChildren<OVRHand>(true);
            if (_requireOvrHands)
            {
                Assert.IsNotNull(cachedValue);
            }

            return cachedValue;
        }

        private void HandleInputDataDirtied(bool isLateUpdate)
        {
            WhenInputDataDirtied(isLateUpdate);
        }

        #region Inject
        public void InjectAllOVRCameraRigRef(InteractionOVRCameraRig ovrCameraRig, bool requireHands)
        {
            InjectInteractionOVRCameraRig(ovrCameraRig);
            InjectRequireHands(requireHands);
        }

        public void InjectInteractionOVRCameraRig(InteractionOVRCameraRig ovrCameraRig)
        {
            _ovrCameraRig = ovrCameraRig;
            // Clear the cached values to force new values to be read on next access
            _leftHand = null;
            _rightHand = null;
        }

        public void InjectRequireHands(bool requireHands)
        {
            _requireOvrHands = requireHands;
        }
        #endregion
    }
}
