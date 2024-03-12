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
using Oculus.Interaction.Throw;

namespace Oculus.Interaction
{
    public class GrabInteractor : PointerInteractor<GrabInteractor, GrabInteractable>, IRigidbodyRef
    {
        [SerializeField, Interface(typeof(ISelector))]
        private MonoBehaviour _selector;

        [SerializeField]
        private Rigidbody _rigidbody;
        public Rigidbody Rigidbody => _rigidbody;

        [SerializeField, Optional]
        private Transform _grabCenter;

        [SerializeField, Optional]
        private Transform _grabTarget;

        private Collider[] _colliders;

        private Tween _tween;

        public float BestInteractableWeight { get; private set; } = float.MaxValue;

        [SerializeField, Interface(typeof(IVelocityCalculator)), Optional]
        private MonoBehaviour _velocityCalculator;
        public IVelocityCalculator VelocityCalculator { get; set; }

        protected override void Awake()
        {
            base.Awake();
            Selector = _selector as ISelector;
            VelocityCalculator = _velocityCalculator as IVelocityCalculator;
        }

        protected override void Start()
        {
            this.BeginStart(ref _started, base.Start);
            Assert.IsNotNull(Selector);
            Assert.IsNotNull(Rigidbody);

            _colliders = Rigidbody.GetComponentsInChildren<Collider>();
            Assert.IsTrue(_colliders.Length > 0,
            "The associated Rigidbody must have at least one Collider.");
            foreach (Collider collider in _colliders)
            {
                Assert.IsTrue(collider.isTrigger,
                    "Associated Colliders must be marked as Triggers.");
            }

            if (_grabCenter == null)
            {
                _grabCenter = transform;
            }

            if (_grabTarget == null)
            {
                _grabTarget = _grabCenter;
            }

            if (_velocityCalculator != null)
            {
                Assert.IsNotNull(VelocityCalculator);
            }

            _tween = new Tween(Pose.identity);

            this.EndStart(ref _started);
        }

        protected override void DoPreprocess()
        {
            transform.position = _grabCenter.position;
            transform.rotation = _grabCenter.rotation;
        }

        protected override GrabInteractable ComputeCandidate()
        {
            GrabInteractable closestInteractable = null;
            float bestScore = float.NegativeInfinity;
            float score = bestScore;

            IEnumerable<GrabInteractable> interactables = GrabInteractable.Registry.List(this);
            foreach (GrabInteractable interactable in interactables)
            {
                Collider[] colliders = interactable.Colliders;
                foreach (Collider collider in colliders)
                {
                    if (Collisions.IsPointWithinCollider(Rigidbody.transform.position, collider))
                    {
                        // Points within a collider are always weighted better than those outside
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

            BestInteractableWeight = bestScore;
            return closestInteractable;
        }

        protected override void InteractableSelected(GrabInteractable interactable)
        {
            Pose target = _grabTarget.GetPose();
            Pose source = _interactable.GetGrabSourceForTarget(target);

            _tween.StopAndSetPose(source);
            base.InteractableSelected(interactable);

            _tween.MoveTo(target);
        }

        protected override void InteractableUnselected(GrabInteractable interactable)
        {
            base.InteractableUnselected(interactable);

            ReleaseVelocityInformation throwVelocity = VelocityCalculator != null ?
                VelocityCalculator.CalculateThrowVelocity(interactable.transform) :
                new ReleaseVelocityInformation(Vector3.zero, Vector3.zero, Vector3.zero);
            interactable.ApplyVelocities(throwVelocity.LinearVelocity, throwVelocity.AngularVelocity);
        }

        protected override void HandlePointerEventRaised(PointerArgs args)
        {
            base.HandlePointerEventRaised(args);

            if (SelectedInteractable == null)
            {
                return;
            }

            if (args.PointerEvent == PointerEvent.Select ||
                args.PointerEvent == PointerEvent.Unselect ||
                args.PointerEvent == PointerEvent.Cancel)
            {
                Pose target = _grabTarget.GetPose();
                if (SelectedInteractable.ResetGrabOnGrabsUpdated)
                {
                    Pose source = _interactable.GetGrabSourceForTarget(target);
                    _tween.StopAndSetPose(source);
                    SelectedInteractable.PointableElement.ProcessPointerEvent(
                        new PointerArgs(Identifier, PointerEvent.Move, _tween.Pose));
                    _tween.MoveTo(target);
                }
                else
                {
                    _tween.StopAndSetPose(target);
                    SelectedInteractable.PointableElement.ProcessPointerEvent(
                        new PointerArgs(Identifier, PointerEvent.Move, target));
                    _tween.MoveTo(target);
                }
            }
        }

        protected override Pose ComputePointerPose()
        {
            if (SelectedInteractable != null)
            {
                return _tween.Pose;
            }
            return _grabTarget.GetPose();
        }

        private bool _outsideReleaseDist = false;

        protected override void DoSelectUpdate()
        {
            GrabInteractable interactable = _selectedInteractable;
            if(interactable == null)
            {
                return;
            }

            _tween.UpdateTarget(_grabTarget.GetPose());
            _tween.Tick();

            _outsideReleaseDist = false;
            if (interactable.ReleaseDistance > 0.0f)
            {
                float closestSqrDist = float.MaxValue;
                Collider[] colliders = interactable.Colliders;
                foreach (Collider collider in colliders)
                {
                    float sqrDistanceFromCenter =
                        (collider.bounds.center - Rigidbody.transform.position).sqrMagnitude;
                    closestSqrDist = Mathf.Min(closestSqrDist, sqrDistanceFromCenter);
                }

                float sqrReleaseDistance = interactable.ReleaseDistance * interactable.ReleaseDistance;

                if (closestSqrDist > sqrReleaseDistance)
                {
                    _outsideReleaseDist = true;
                }
            }
        }

        public override bool ShouldUnselect {
            get
            {
                if (State != InteractorState.Select)
                {
                    return false;
                }

                if (_outsideReleaseDist)
                {
                    return true;
                }

                return base.ShouldUnselect;
            }
        }

        #region Inject
        public void InjectAllGrabInteractor(ISelector selector, Rigidbody rigidbody)
        {
            InjectSelector(selector);
            InjectRigidbody(rigidbody);
        }

        public void InjectSelector(ISelector selector)
        {
            _selector = selector as MonoBehaviour;
            Selector = selector;
        }

        public void InjectRigidbody(Rigidbody rigidbody)
        {
            _rigidbody = rigidbody;
        }

        public void InjectOptionalGrabCenter(Transform grabCenter)
        {
            _grabCenter = grabCenter;
        }

        public void InjectOptionalGrabTarget(Transform grabTarget)
        {
            _grabTarget = grabTarget;
        }

        public void InjectOptionalVelocityCalculator(IVelocityCalculator velocityCalculator)
        {
            _velocityCalculator = velocityCalculator as MonoBehaviour;
            VelocityCalculator = velocityCalculator;
        }

        #endregion
    }
}
