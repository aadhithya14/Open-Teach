/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Serialization;

namespace Oculus.Interaction
{
    /// <summary>
    /// The SnapInteractor referes to an element that can snap to a SnapInteractable.
    /// This interactor moves itself into the Pose specified by the intereactable.
    /// Additionally, it can specify a preferred SnapInteractable and a TimeOut time, and it
    /// will automatically snap there if its Pointable element has not been used (hovered, selected)
    /// for a certain time.
    /// </summary>
    public class SnapInteractor : Interactor<SnapInteractor, SnapInteractable>,
        IRigidbodyRef
    {
        [SerializeField]
        private PointableElement _pointableElement;

        [SerializeField]
        private Rigidbody _rigidbody;
        public Rigidbody Rigidbody => _rigidbody;

        [SerializeField, Optional]
        [FormerlySerializedAs("_dropPoint")]
        private Transform _snapPoint;
        public Pose SnapPose => _snapPoint.GetPose();

        [Header("Time out")]
        [SerializeField, Optional]
        private SnapInteractable _timeOutInteractable;
        [SerializeField, Optional]
        private float _timeOut = 0f;

        private float _idleStarted = -1f;
        private IMovement _movement;

        #region Editor events
        private void Reset()
        {
            _rigidbody = this.GetComponentInParent<Rigidbody>();
            _pointableElement = this.GetComponentInParent<PointableElement>();
        }
        #endregion

        #region Unity Lifecycle
        protected override void Awake()
        {
            base.Awake();
            _pointableElement = _pointableElement as PointableElement;
        }

        protected override void Start()
        {
            this.BeginStart(ref _started, base.Start);
            Assert.IsNotNull(_pointableElement, "Pointable element can not be null");
            Assert.IsNotNull(Rigidbody, "Rigidbody can not be null");
            if (_snapPoint == null)
            {
                _snapPoint = this.transform;
            }

            this.EndStart(ref _started);
        }

        protected override void OnEnable()
        {
            base.OnEnable();
            if (_started)
            {
                _pointableElement.WhenPointerEventRaised += HandlePointerEventRaised;
            }
        }

        protected override void OnDisable()
        {
            if (_started)
            {
                _pointableElement.WhenPointerEventRaised -= HandlePointerEventRaised;
            }
            base.OnDisable();
        }

        #endregion

        #region Interactor Lifecycle

        public override bool ShouldSelect
        {
            get
            {
                if (State != InteractorState.Hover)
                {
                    return false;
                }

                return _pointableElement.SelectingPointsCount == 0;
            }
        }

        public override bool ShouldUnselect {
            get
            {
                if (State != InteractorState.Select)
                {
                    return false;
                }

                return _shouldUnselect;
            }
        }


        protected override void DoHoverUpdate()
        {
            base.DoHoverUpdate();

            if (Interactable == null)
            {
                return;
            }

            Interactable.InteractorHoverUpdated(this);
        }

        private bool _shouldUnselect = false;

        protected override void DoSelectUpdate()
        {
            base.DoSelectUpdate();

            _shouldUnselect = false;

            if (_movement == null)
            {
                _shouldUnselect = true;
                return;
            }

            if (Interactable != null)
            {
                if (Interactable.PoseForInteractor(this, out Pose targetPose))
                {
                    _movement.UpdateTarget(targetPose);
                    _movement.Tick();
                    GeneratePointerEvent(PointerEvent.Move);
                }
                else
                {
                    _shouldUnselect = true;
                }
            }
            else
            {
                _shouldUnselect = true;
            }

            if (_pointableElement.SelectingPointsCount > 1)
            {
                _shouldUnselect = true;
            }
        }

        protected override void InteractableSet(SnapInteractable interactable)
        {
            base.InteractableSet(interactable);
            if (interactable != null)
            {
                GeneratePointerEvent(PointerEvent.Hover);
            }
        }

        protected override void InteractableUnset(SnapInteractable interactable)
        {
            if (interactable != null)
            {
                GeneratePointerEvent(PointerEvent.Unhover);
            }
            base.InteractableUnset(interactable);
        }

        protected override void InteractableSelected(SnapInteractable interactable)
        {
            base.InteractableSelected(interactable);
            if (interactable != null)
            {
                _movement = interactable.GenerateMovement(_snapPoint.GetPose(), this);
                if (_movement != null)
                {
                    GeneratePointerEvent(PointerEvent.Select);
                }
            }
        }

        protected override void InteractableUnselected(SnapInteractable interactable)
        {
            _movement?.StopAndSetPose(_movement.Pose);
            if (interactable != null)
            {
                GeneratePointerEvent(PointerEvent.Unselect);
            }
            base.InteractableUnselected(interactable);
            _movement = null;
        }

        #endregion

        #region Pointable

        protected virtual void HandlePointerEventRaised(PointerArgs args)
        {
            if (_pointableElement.PointsCount == 0
                && (args.PointerEvent == PointerEvent.Cancel
                    || args.PointerEvent == PointerEvent.Unhover
                    || args.PointerEvent == PointerEvent.Unselect))
            {
                _idleStarted = Time.time;
            }
            else
            {
                _idleStarted = -1f;
            }

            if (args.Identifier == Identifier
                && args.PointerEvent == PointerEvent.Cancel
                && Interactable != null)
            {
                Interactable.RemoveInteractorById(Identifier);
            }
        }


        public void GeneratePointerEvent(PointerEvent pointerEvent, Pose pose)
        {
            _pointableElement.ProcessPointerEvent(new PointerArgs(Identifier, pointerEvent, pose));
        }

        private void GeneratePointerEvent(PointerEvent pointerEvent)
        {
            Pose pose = ComputePointerPose();
            _pointableElement.ProcessPointerEvent(new PointerArgs(Identifier, pointerEvent, pose));
        }

        protected Pose ComputePointerPose()
        {
            if (_movement != null)
            {
                return _movement.Pose;
            }

            return SnapPose;
        }
        #endregion

        private bool TimedOut()
        {
            return _timeOut >= 0f
                && _idleStarted >= 0f
                && Time.time - _idleStarted > _timeOut;
        }

        protected override SnapInteractable ComputeCandidate()
        {
            SnapInteractable interactable = ComputeIntersectingCandidate();
            if (TimedOut())
            {
                return interactable != null ? interactable : _timeOutInteractable;
            }
            return interactable;
        }

        private SnapInteractable ComputeIntersectingCandidate()
        {
            SnapInteractable closestInteractable = null;
            float bestScore = float.MinValue;
            float score;

            IEnumerable<SnapInteractable> interactables = SnapInteractable.Registry.List(this);
            foreach (SnapInteractable interactable in interactables)
            {
                Collider[] colliders = interactable.Colliders;
                foreach (Collider collider in colliders)
                {
                    if (Collisions.IsPointWithinCollider(Rigidbody.transform.position, collider))
                    {
                        float sqrDistanceFromCenter =
                            (Rigidbody.transform.position - collider.bounds.center).magnitude;
                        score = float.MaxValue - sqrDistanceFromCenter;
                    }
                    else
                    {
                        var position = Rigidbody.transform.position;
                        Vector3 closestPointOnInteractable = collider.ClosestPoint(position);
                        score = -1f * (position - closestPointOnInteractable).magnitude;
                    }

                    if (score > bestScore)
                    {
                        bestScore = score;
                        closestInteractable = interactable;
                    }
                }
            }
            return closestInteractable;
        }

        #region Inject

        public void InjectAllSnapInteractor(PointableElement pointableElement, Rigidbody rigidbody)
        {
            InjectPointableElement(pointableElement);
            InjectRigidbody(rigidbody);
        }

        public void InjectPointableElement(PointableElement pointableElement)
        {
            _pointableElement = pointableElement;
        }

        public void InjectRigidbody(Rigidbody rigidbody)
        {
            _rigidbody = rigidbody;
        }

        public void InjectOptionalSnapPoint(Transform snapPoint)
        {
            _snapPoint = snapPoint;
        }

        public void InjectOptionalTimeOutInteractable(SnapInteractable interactable)
        {
            _timeOutInteractable = interactable;
        }

        public void InjectOptionaTimeOut(float timeOut)
        {
            _timeOut = timeOut;
        }
        #endregion
    }
}
