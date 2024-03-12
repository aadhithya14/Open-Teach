/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using Oculus.Interaction.Grab;
using Oculus.Interaction.GrabAPI;
using Oculus.Interaction.Input;
using Oculus.Interaction.Throw;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

namespace Oculus.Interaction.HandGrab
{
    /// <summary>
    /// The DistanceHandGrabInteractor allows grabbing DistanceHandGrabInteractables at a distance.
    /// Similarly to the HandGrabInteractor it operates with HandGrabPoses to specify the final pose of the hand
    /// and as well as attracting objects at a distance it will held them in the same manner the HandGrabInteractor does.
    /// The DistanceHandGrabInteractor does not need a collider and uses conical frustums to detect far-away objects.
    /// </summary>
    public class DistanceHandGrabInteractor :
        PointerInteractor<DistanceHandGrabInteractor, DistanceHandGrabInteractable>
        , IHandGrabState, IHandGrabber, IDistanceInteractor
    {
        [SerializeField, Interface(typeof(IHand))]
        private MonoBehaviour _hand;
        public IHand Hand { get; private set; }

        [SerializeField]
        private HandGrabAPI _handGrabApi;

        [SerializeField]
        private Transform _grabOrigin;

        [Header("Distance selection volumes")]
        [SerializeField]
        private DistantPointDetectorFrustums _detectionFrustums;

        [Header("Grabbing")]
        [SerializeField]
        private GrabTypeFlags _supportedGrabTypes = GrabTypeFlags.Pinch;

        [SerializeField, Optional]
        private Transform _gripPoint;

        [SerializeField, Optional]
        private Transform _pinchPoint;

        [SerializeField]
        private float _detectionDelay = 0f;

        [SerializeField, Interface(typeof(IVelocityCalculator)), Optional]
        private MonoBehaviour _velocityCalculator;

        public IVelocityCalculator VelocityCalculator { get; set; }

        private HandGrabTarget _currentTarget = new HandGrabTarget();
        private HandPose _cachedBestHandPose = new HandPose();
        private Pose _cachedBestGrabPose = Pose.identity;

        private IMovement _movement;

        private HandGrabTarget _immediateTarget = new HandGrabTarget();
        private DistanceHandGrabInteractable _snappyCandidate;
        private DistanceHandGrabInteractable _inmediateCandidate;
        private float _hoverStartTime;

        private Pose _wristToGrabAnchorOffset = Pose.identity;
        private Pose _grabPose = Pose.identity;
        private Pose _gripPose = Pose.identity;
        private Pose _pinchPose = Pose.identity;

        private bool _handGrabShouldSelect = false;
        private bool _handGrabShouldUnselect = false;

        private HandGrabbableData _lastInteractableData =
            new HandGrabbableData();

        #region IHandGrabber

        public HandGrabAPI HandGrabApi => _handGrabApi;
        public GrabTypeFlags SupportedGrabTypes => _supportedGrabTypes;
        public IHandGrabbable TargetInteractable => Interactable;

        #endregion

        public ConicalFrustum PointerFrustum => _detectionFrustums.SelectionFrustum;

        #region IHandGrabSource

        public virtual bool IsGrabbing => HasSelectedInteractable
            && (_movement == null || _movement.Stopped);

        public float GrabStrength { get; private set; }
        public Pose WristToGrabPoseOffset => _wristToGrabAnchorOffset;

        public HandFingerFlags GrabbingFingers() =>
            Grab.HandGrab.GrabbingFingers(this, SelectedInteractable);

        public HandGrabTarget HandGrabTarget { get; private set; }
        public System.Action<IHandGrabState> WhenHandGrabStarted { get; set; } = delegate { };
        public System.Action<IHandGrabState> WhenHandGrabEnded { get; set; } = delegate { };

        #endregion

        private DistantPointDetector _detector;

        #region editor events

        protected virtual void Reset()
        {
            _hand = this.GetComponentInParent<IHand>() as MonoBehaviour;
            _handGrabApi = this.GetComponentInParent<HandGrabAPI>();
        }

        #endregion

        protected override void Awake()
        {
            base.Awake();
            Hand = _hand as IHand;
            VelocityCalculator = _velocityCalculator as IVelocityCalculator;
        }

        protected override void Start()
        {
            this.BeginStart(ref _started, base.Start);
            Assert.IsNotNull(Hand, "Hand can not be null");
            Assert.IsNotNull(_handGrabApi, "HandGrabAPI can not be null");
            Assert.IsNotNull(_grabOrigin);
            Assert.IsNotNull(PointerFrustum, "The selector frustum can not be null");
            if (_velocityCalculator != null)
            {
                Assert.IsNotNull(VelocityCalculator, "The provided Velocity Calculator is not an IVelocityCalculator");
            }

            _detector = new DistantPointDetector(_detectionFrustums);
            this.EndStart(ref _started);
        }

        #region life cycle

        protected override void DoPreprocess()
        {
            base.DoPreprocess();

            _grabPose = _grabOrigin.GetPose();

            if (Hand.Handedness == Handedness.Left)
            {
                _grabPose.rotation *= Quaternion.Euler(180f, 0f, 0f);
            }

            if (_gripPoint != null)
            {
                _gripPose = _gripPoint.GetPose();
            }
            if (_pinchPoint != null)
            {
                _pinchPose = _pinchPoint.GetPose();
            }
        }

        public override bool ShouldSelect
        {
            get
            {
                if (State != InteractorState.Hover)
                {
                    return false;
                }

                return _candidate == _interactable && _handGrabShouldSelect;
            }
        }

        public override bool ShouldUnselect
        {
            get
            {
                if (State != InteractorState.Select)
                {
                    return false;
                }

                return _handGrabShouldUnselect;
            }
        }

        protected override void DoHoverUpdate()
        {
            base.DoHoverUpdate();

            if (Interactable != null)
            {
                _handGrabShouldSelect = false;
                _wristToGrabAnchorOffset = GetAnchorOffset(_currentTarget.Anchor);
                GrabStrength = Grab.HandGrab.ComputeHandGrabScore(this, Interactable,
                    out GrabTypeFlags hoverGrabTypes);
                HandGrabTarget = _currentTarget;
            }
            else
            {
                _wristToGrabAnchorOffset = Pose.identity;
                GrabStrength = 0f;
                HandGrabTarget = null;
            }

            if (Interactable != null
                && Grab.HandGrab.ComputeShouldSelect(this, Interactable, out GrabTypeFlags selectingGrabTypes))
            {
                _handGrabShouldSelect = true;
            }
        }

        protected override void DoSelectUpdate()
        {
            DistanceHandGrabInteractable interactable = _selectedInteractable;
            _handGrabShouldUnselect = false;
            if (interactable == null)
            {
                _currentTarget.Clear();
                _handGrabShouldUnselect = true;
                return;
            }

            Pose grabPose = PoseUtils.Multiply(_grabPose, _wristToGrabAnchorOffset);
            _movement.UpdateTarget(grabPose);
            _movement.Tick();

            Grab.HandGrab.StoreGrabData(this, interactable, ref _lastInteractableData);
            if (Grab.HandGrab.ComputeShouldUnselect(this, interactable))
            {
                _handGrabShouldUnselect = true;
            }
        }

        protected override void InteractableSelected(DistanceHandGrabInteractable interactable)
        {
            if (interactable == null)
            {
                base.InteractableSelected(interactable);
                return;
            }

            _wristToGrabAnchorOffset = GetAnchorOffset(_currentTarget.Anchor);

            Pose grabPose = PoseUtils.Multiply(_grabPose, _wristToGrabAnchorOffset);
            Pose interactableGrabStartPose = _currentTarget.WorldGrabPose;
            _movement = interactable.GenerateMovement(interactableGrabStartPose, grabPose);
            base.InteractableSelected(interactable);
            interactable.WhenPointerEventRaised += HandleOtherPointerEventRaised;
        }

        protected override void InteractableUnselected(DistanceHandGrabInteractable interactable)
        {
            interactable.WhenPointerEventRaised -= HandleOtherPointerEventRaised;
            _movement?.StopAndSetPose(_movement.Pose);
            base.InteractableUnselected(interactable);
            _movement = null;

            ReleaseVelocityInformation throwVelocity = VelocityCalculator != null ?
                VelocityCalculator.CalculateThrowVelocity(interactable.transform) :
                new ReleaseVelocityInformation(Vector3.zero, Vector3.zero, Vector3.zero);
            interactable.ApplyVelocities(throwVelocity.LinearVelocity, throwVelocity.AngularVelocity);
        }

        protected override void InteractableSet(DistanceHandGrabInteractable interactable)
        {
            base.InteractableSet(interactable);
            WhenHandGrabStarted.Invoke(this);
        }

        protected override void InteractableUnset(DistanceHandGrabInteractable interactable)
        {
            base.InteractableUnset(interactable);
            WhenHandGrabEnded.Invoke(this);
        }

        protected override Pose ComputePointerPose()
        {
            if (SelectedInteractable != null)
            {
                return _movement.Pose;
            }

            return _grabPose;
        }

        protected virtual void HandleOtherPointerEventRaised(PointerArgs args)
        {
            if (SelectedInteractable == null)
            {
                return;
            }

            if (args.Identifier != Identifier &&
                (args.PointerEvent == PointerEvent.Select || args.PointerEvent == PointerEvent.Unselect))
            {
                Pose grabPose = PoseUtils.Multiply(_grabPose, _wristToGrabAnchorOffset);
                if (SelectedInteractable.ResetGrabOnGrabsUpdated)
                {
                    if (SelectedInteractable.CalculateBestPose(grabPose, Hand.Scale, Hand.Handedness,
                        ref _cachedBestHandPose, ref _cachedBestGrabPose,
                        out bool usesHandPose, out float poseScore))
                    {
                        HandGrabTarget.GrabAnchor anchor = _currentTarget.Anchor;
                        HandPose handPose = usesHandPose ? _cachedBestHandPose : null;
                        _currentTarget.Set(SelectedInteractable.RelativeTo, SelectedInteractable.HandAlignment, handPose, _cachedBestGrabPose, anchor);
                    }
                }

                Pose fromPose = _currentTarget.WorldGrabPose;
                _movement = SelectedInteractable.GenerateMovement(fromPose, grabPose);
                SelectedInteractable.PointableElement.ProcessPointerEvent(
                    new PointerArgs(Identifier, PointerEvent.Move, fromPose));
            }
        }

        #endregion

        private HandGrabTarget.GrabAnchor AnchorMode(DistanceHandGrabInteractable interactable, GrabTypeFlags grabTypes)
        {
            if (interactable != null && interactable.UsesHandPose())
            {
                return HandGrabTarget.GrabAnchor.Wrist;
            }
            if (_pinchPoint != null && (grabTypes & GrabTypeFlags.Pinch) != 0)
            {
                return HandGrabTarget.GrabAnchor.Pinch;
            }
            if (_gripPoint != null)
            {
                return HandGrabTarget.GrabAnchor.Palm;
            }

            return HandGrabTarget.GrabAnchor.Wrist;
        }

        private Pose GetAnchorOffset(HandGrabTarget.GrabAnchor anchor)
        {
            if (anchor == HandGrabTarget.GrabAnchor.Pinch)
            {
                return PoseUtils.RelativeOffset(_pinchPose, _grabPose);
            }
            else if (anchor == HandGrabTarget.GrabAnchor.Palm)
            {
                return PoseUtils.RelativeOffset(_gripPose, _grabPose);
            }

            return Pose.identity;
        }

        protected override DistanceHandGrabInteractable ComputeCandidate()
        {
            if (_snappyCandidate != null
                && _detector.IsPointingWithoutAid(_snappyCandidate.Colliders))
            {
                return _snappyCandidate;
            }

            if (_snappyCandidate != null
                && !_detector.ComputeIsPointing(_snappyCandidate.Colliders, false,
                        out float score, out Vector3 bestHitPoint))
            {
                _currentTarget.Clear();
                _snappyCandidate = null;
            }

            DistanceHandGrabInteractable candidate = ComputeBestHandGrabTarget(ref _immediateTarget, _snappyCandidate == null);
            if (candidate != _inmediateCandidate)
            {
                _inmediateCandidate = candidate;
                if (candidate != null)
                {
                    _hoverStartTime = Time.time;
                }
            }

            if (_snappyCandidate == null
                || (candidate != null && _snappyCandidate != candidate && Time.time - _hoverStartTime >= _detectionDelay))
            {
                _inmediateCandidate = null;
                _snappyCandidate = candidate;
                _currentTarget.Set(_immediateTarget);
            }

            return _snappyCandidate;
        }

        protected DistanceHandGrabInteractable ComputeBestHandGrabTarget(ref HandGrabTarget handGrabTarget, bool wideSearch)
        {
            DistanceHandGrabInteractable closestInteractable = null;
            float bestScore = float.NegativeInfinity;
            float bestFingerScore = float.NegativeInfinity;

            IEnumerable<DistanceHandGrabInteractable> interactables = DistanceHandGrabInteractable.Registry.List(this);

            foreach (DistanceHandGrabInteractable interactable in interactables)
            {
                float fingerScore = 1.0f;
                if (!Grab.HandGrab.ComputeShouldSelect(this, interactable, out GrabTypeFlags selectingGrabTypes))
                {
                    fingerScore = Grab.HandGrab.ComputeHandGrabScore(this, interactable, out selectingGrabTypes);
                    if (selectingGrabTypes == GrabTypeFlags.None)
                    {
                        selectingGrabTypes = _supportedGrabTypes;
                    }
                }
                if (fingerScore < bestFingerScore)
                {
                    continue;
                }

                if (!_detector.ComputeIsPointing(interactable.Colliders, !wideSearch, out float score, out Vector3 hitPoint)
                    || score < bestScore)
                {
                    continue;
                }

                HandGrabTarget.GrabAnchor anchorMode = AnchorMode(interactable, selectingGrabTypes);
                Pose grabPose = anchorMode == HandGrabTarget.GrabAnchor.Pinch ? _pinchPose :
                    anchorMode == HandGrabTarget.GrabAnchor.Palm ? _gripPose :
                    _grabPose;

                Pose worldPose = new Pose(hitPoint, grabPose.rotation);
                bool poseFound = interactable.CalculateBestPose(worldPose, Hand.Scale, Hand.Handedness,
                    ref _cachedBestHandPose, ref _cachedBestGrabPose,
                    out bool usesHandPose, out float poseScore);

                if (!poseFound)
                {
                    continue;
                }

                bestScore = score;
                closestInteractable = interactable;
                HandPose handPose = usesHandPose ? _cachedBestHandPose : null;
                handGrabTarget.Set(interactable.RelativeTo, interactable.HandAlignment, handPose, _cachedBestGrabPose, anchorMode);
            }

            return closestInteractable;
        }

        #region Inject
        public void InjectAllDistanceHandGrabInteractor(HandGrabAPI handGrabApi,
            DistantPointDetectorFrustums frustums,
            Transform grabOrigin,
            IHand hand, GrabTypeFlags supportedGrabTypes)
        {
            InjectHandGrabApi(handGrabApi);
            InjectDetectionFrustums(frustums);
            InjectGrabOrigin(grabOrigin);
            InjectHand(hand);
            InjectSupportedGrabTypes(supportedGrabTypes);
        }

        public void InjectHandGrabApi(HandGrabAPI handGrabApi)
        {
            _handGrabApi = handGrabApi;
        }

        public void InjectDetectionFrustums(DistantPointDetectorFrustums frustums)
        {
            _detectionFrustums = frustums;
        }

        public void InjectGrabOrigin(Transform grabOrigin)
        {
            _grabOrigin = grabOrigin;
        }

        public void InjectHand(IHand hand)
        {
            _hand = hand as MonoBehaviour;
            Hand = hand;
        }

        public void InjectSupportedGrabTypes(GrabTypeFlags supportedGrabTypes)
        {
            _supportedGrabTypes = supportedGrabTypes;
        }

        public void InjectOptionalGripPoint(Transform gripPoint)
        {
            _gripPoint = gripPoint;
        }

        public void InjectOptionalPinchPoint(Transform pinchPoint)
        {
            _pinchPoint = pinchPoint;
        }

        public void InjectOptionalVelocityCalculator(IVelocityCalculator velocityCalculator)
        {
            _velocityCalculator = velocityCalculator as MonoBehaviour;
            VelocityCalculator = velocityCalculator;
        }
        #endregion
    }
}
