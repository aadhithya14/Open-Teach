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
using Oculus.Interaction.Surfaces;

namespace Oculus.Interaction
{
    /// <summary>
    /// Defines a near-poke interaction that is driven by a near-distance
    /// proximity computation and a raycast between the position
    /// recorded across two frames against a target surface.
    /// </summary>
    public class PokeInteractor : PointerInteractor<PokeInteractor, PokeInteractable>
    {
        [SerializeField]
        private Transform _pointTransform;

        [SerializeField]
        private float _touchReleaseThreshold = 0.002f;

        [SerializeField]
        private ProgressCurve _dragStartCurve;

        public Vector3 ClosestPoint { get; private set; }

        public Vector3 TouchPoint { get; private set; }

        public Vector3 Origin { get; private set; }

        private Vector3 _previousOrigin;
        private Vector3 _previousTouchPoint;
        private Vector3 _capturedTouchPoint;
        private Vector3 _startDragOffset;

        private PokeInteractable _previousCandidate = null;
        private PokeInteractable _hitInteractable = null;

        private bool _dragging;

        protected override void Start()
        {
            base.Start();
            Assert.IsNotNull(_pointTransform);
        }

        protected override void DoPreprocess()
        {
            _previousOrigin = Origin;
            Origin = _pointTransform.position;
        }

        public override bool ShouldSelect
        {
            get
            {
                if (State != InteractorState.Hover)
                {
                    return false;
                }

                return _candidate == _interactable && _hitInteractable != null;
            }
        }

        public override bool ShouldUnselect {
            get
            {
                if (State != InteractorState.Select)
                {
                    return false;
                }

                return _hitInteractable == null;
            }
        }

        protected override void DoHoverUpdate()
        {
            if (_interactable != null)
            {
                TouchPoint = _interactable.ComputeClosestPoint(Origin);
            }
        }

        protected override PokeInteractable ComputeCandidate()
        {
            _hitInteractable = null;

            // First, see if we trigger a press on any interactable
            PokeInteractable closestInteractable = ComputeSelectCandidate();
            if (closestInteractable != null)
            {
                // We have found an active hit target, so we return it
                _hitInteractable = closestInteractable;
                _previousCandidate = closestInteractable;
                return _hitInteractable;
            }

            // Otherwise we have no active interactable, so we do a proximity-only check for
            // closest hovered interactable (above the surface)
            closestInteractable = ComputeBestHoverInteractable();
            _previousCandidate = closestInteractable;

            return closestInteractable;
        }

        private PokeInteractable ComputeSelectCandidate()
        {
            PokeInteractable closestInteractable = null;
            float closestSqrDist = float.MaxValue;

            IEnumerable<PokeInteractable> interactables = PokeInteractable.Registry.List(this);

            Vector3 moveDirection = Origin - _previousOrigin;
            float magnitude = moveDirection.magnitude;
            if (magnitude == 0f)
            {
                return null;
            }

            moveDirection /= magnitude;
            Ray ray = new Ray(_previousOrigin, moveDirection);

            // Check the surface first as a movement through this will
            // automatically put us in a "active" state. We expect the raycast
            // to happen only in one direction
            foreach (PokeInteractable interactable in interactables)
            {
                if (!PassesEnterHoverDistanceCheck(interactable))
                {
                    continue;
                }

                Vector3 closestSurfaceNormal = interactable.ClosestSurfaceNormal(Origin);

                // First check that we are moving towards the surface by checking
                // the direction of our position delta with the forward direction of the surface normal.
                // This is to not allow presses from "behind" the surface.

                // Check if we are moving toward the surface
                if (Vector3.Dot(moveDirection, closestSurfaceNormal) < 0f)
                {
                    // Then do a raycast against the surface
                    bool hit = interactable.Surface.Raycast(ray, out SurfaceHit surfaceHit);
                    if (hit && surfaceHit.Distance <= magnitude)
                    {
                        // We collided against the surface and now we must rank this
                        // interactable versus others that also pass this test this frame
                        // but may be at a closer proximity. For this we use the closest
                        // point compute against the surface intersection point

                        // Check if our collision lies outside of the optional volume mask
                        if (interactable.VolumeMask != null &&
                            !Collisions.IsPointWithinCollider(surfaceHit.Point, interactable.VolumeMask))
                        {
                            continue;
                        }

                        Vector3 closestPointToHitPoint = interactable.ComputeClosestPoint(surfaceHit.Point);

                        float sqrDistanceFromPoint = (closestPointToHitPoint - surfaceHit.Point).sqrMagnitude;

                        if (sqrDistanceFromPoint > interactable.MaxDistance * interactable.MaxDistance)
                        {
                            continue;
                        }

                        if (sqrDistanceFromPoint < closestSqrDist)
                        {
                            closestSqrDist = sqrDistanceFromPoint;
                            closestInteractable = interactable;
                            ClosestPoint = closestPointToHitPoint;
                            TouchPoint = ClosestPoint;
                        }
                    }
                }
            }
            return closestInteractable;
        }

        private bool PassesEnterHoverDistanceCheck(PokeInteractable interactable)
        {
            if (interactable == _previousCandidate)
            {
                return true;
            }

            if(ComputeDistanceAbove(interactable, _previousOrigin) > -1f * interactable.EnterHoverDistance)
            {
                return false;
            }

            return true;
        }

        private PokeInteractable ComputeBestHoverInteractable()
        {
            PokeInteractable closestInteractable = null;
            float closestSqrDist = float.MaxValue;

            IEnumerable<PokeInteractable> interactables = PokeInteractable.Registry.List(this);

            // We check that we're above the surface first as we don't
            // care about hovers that originate below the surface
            foreach (PokeInteractable interactable in interactables)
            {
                if (!PassesEnterHoverDistanceCheck(interactable))
                {
                    continue;
                }

                Vector3 closestSurfacePoint = interactable.ClosestSurfacePoint(Origin);
                Vector3 closestSurfaceNormal = interactable.ClosestSurfaceNormal(Origin);

                Vector3 surfaceToPoint = Origin - closestSurfacePoint;
                float magnitude = surfaceToPoint.magnitude;
                if (magnitude != 0f)
                {
                    // Check if our position is above the surface
                    if (Vector3.Dot(surfaceToPoint, closestSurfaceNormal) > 0f)
                    {
                        // Check if our position lies outside of the optional volume mask
                        if (interactable.VolumeMask != null &&
                            !Collisions.IsPointWithinCollider(Origin, interactable.VolumeMask))
                        {
                            continue;
                        }

                        // We're above the surface so now we must rank this
                        // interactable versus others that also pass this test this frame
                        // but may be at a closer proximity.
                        Vector3 closestPoint = interactable.ComputeClosestPoint(Origin);

                        float sqrDistanceFromPoint = (closestPoint - Origin).sqrMagnitude;

                        if (sqrDistanceFromPoint > interactable.MaxDistance * interactable.MaxDistance)
                        {
                            continue;
                        }

                        if (sqrDistanceFromPoint < closestSqrDist)
                        {
                            closestSqrDist = sqrDistanceFromPoint;
                            closestInteractable = interactable;
                            ClosestPoint = closestPoint;
                            TouchPoint = ClosestPoint;
                        }
                    }
                }
            }
            return closestInteractable;
        }

        protected override void InteractableSelected(PokeInteractable interactable)
        {
            _dragging = false;
            if (interactable != null)
            {
                Vector3 worldPosition = interactable.ClosestSurfacePoint(Origin);
                _previousTouchPoint = worldPosition;
                _capturedTouchPoint = worldPosition;
            }

            base.InteractableSelected(interactable);
        }

        protected override Pose ComputePointerPose()
        {
            if (Interactable == null)
            {
                return Pose.identity;
            }

            return new Pose(
                TouchPoint,
                Quaternion.LookRotation(Interactable.ClosestSurfaceNormal(TouchPoint))
            );
        }

        private float ComputeDistanceAbove(PokeInteractable interactable, Vector3 point)
        {
            Vector3 closestSurfacePoint = interactable.ClosestSurfacePoint(point);
            Vector3 closestSurfaceNormal = interactable.ClosestSurfaceNormal(point);
            Vector3 surfaceToPoint = point - closestSurfacePoint;
            return Vector3.Dot(surfaceToPoint, -closestSurfaceNormal);
        }

        private float ComputeDepth(PokeInteractable interactable, Vector3 point)
        {
            return Mathf.Max(0f, ComputeDistanceAbove(interactable, point));
        }

        protected override void DoSelectUpdate()
        {
            PokeInteractable interactable = _selectedInteractable;
            if (interactable == null)
            {
                _hitInteractable = null;
                return;
            }

            Vector3 closestSurfacePoint = interactable.ClosestSurfacePoint(Origin);
            Vector3 closestSurfaceNormal = interactable.ClosestSurfaceNormal(Origin);
            Vector3 surfaceToInteractor = Origin - closestSurfacePoint;

            // Unselect our interactor if it is above the surface by at least releaseDistancePadding
            if (Vector3.Dot(surfaceToInteractor, closestSurfaceNormal) > _touchReleaseThreshold)
            {
                _hitInteractable = null;
                return;
            }

            Vector3 worldPositionOnSurface = interactable.ClosestSurfacePoint(Origin);

            Vector2 lateralDelta =
                interactable.Surface.GetSurfaceDistanceBetween(worldPositionOnSurface,
                                                               _capturedTouchPoint);

            Vector2 frameDelta =
                interactable.Surface.GetSurfaceDistanceBetween(worldPositionOnSurface,
                                                               _previousTouchPoint);

            float depthDelta = Mathf.Abs(ComputeDepth(interactable, Origin) -
                                         ComputeDepth(interactable, _previousOrigin));
            bool outsideDelta = false;

            if (!_dragging && frameDelta.magnitude > depthDelta)
            {
                while (!outsideDelta)
                {
                    if (lateralDelta.x > _selectedInteractable.HorizontalDragThreshold)
                    {
                        outsideDelta = true;
                        break;
                    }

                    if (lateralDelta.y > _selectedInteractable.VerticalDragThreshold)
                    {
                        outsideDelta = true;
                        break;
                    }

                    break;
                }

                if (outsideDelta)
                {
                    _dragStartCurve.Start();
                    _startDragOffset = _capturedTouchPoint - worldPositionOnSurface;
                    _dragging = true;
                }
            }

            if (!_dragging)
            {
                TouchPoint = _capturedTouchPoint;
            }
            else
            {
                float deltaEase = _dragStartCurve.Progress();
                Vector3 offset = Vector3.Lerp(_startDragOffset, Vector3.zero, deltaEase);

                TouchPoint = worldPositionOnSurface + offset;
            }

            _previousTouchPoint = worldPositionOnSurface;

            Vector3 closestPoint = interactable.ComputeClosestPoint(Origin);
            float distanceFromPoint = (closestPoint - Origin).magnitude;
            if (interactable.ReleaseDistance > 0.0f)
            {
                if (distanceFromPoint > interactable.ReleaseDistance)
                {
                    GeneratePointerEvent(PointerEvent.Cancel, interactable);
                    _previousCandidate = null;
                    _previousOrigin = Origin;
                    _hitInteractable = null;
                }
            }
        }

        #region Inject

        public void InjectAllPokeInteractor(Transform pointTransform)
        {
            InjectPointTransform(pointTransform);
        }

        public void InjectPointTransform(Transform pointTransform)
        {
            _pointTransform = pointTransform;
        }

        public void InjectOptionalTouchReleaseThreshold(float touchReleaseThreshold)
        {
            _touchReleaseThreshold = touchReleaseThreshold;
        }

        public void InjectOptionDragStartCurve(ProgressCurve dragStartCurve)
        {
            _dragStartCurve = dragStartCurve;
        }

        #endregion
    }
}
