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

namespace Oculus.Interaction
{
    public abstract class PointerInteractor<TInteractor, TInteractable> : Interactor<TInteractor, TInteractable>
                                    where TInteractor : Interactor<TInteractor, TInteractable>
                                    where TInteractable : PointerInteractable<TInteractor, TInteractable>
    {
        protected void GeneratePointerEvent(PointerEvent pointerEvent, TInteractable interactable)
        {
            Pose pose = ComputePointerPose();

            if (interactable == null)
            {
                return;
            }

            if (interactable.PointableElement != null)
            {
                if (pointerEvent == PointerEvent.Hover)
                {
                    interactable.PointableElement.WhenPointerEventRaised +=
                        HandlePointerEventRaised;
                }
                else if (pointerEvent == PointerEvent.Unhover)
                {
                    interactable.PointableElement.WhenPointerEventRaised -=
                        HandlePointerEventRaised;
                }
            }

            interactable.PublishPointerEvent(new PointerArgs(Identifier, pointerEvent, pose));
        }

        protected virtual void HandlePointerEventRaised(PointerArgs args)
        {
            if (args.Identifier == Identifier &&
                args.PointerEvent == PointerEvent.Cancel &&
                Interactable != null)
            {
                TInteractable interactable = Interactable;
                interactable.RemoveInteractorById(Identifier);
                interactable.PointableElement.WhenPointerEventRaised -=
                    HandlePointerEventRaised;
            }
        }

        protected override void InteractableSet(TInteractable interactable)
        {
            base.InteractableSet(interactable);
            GeneratePointerEvent(PointerEvent.Hover, interactable);
        }

        protected override void InteractableUnset(TInteractable interactable)
        {
            GeneratePointerEvent(PointerEvent.Unhover, interactable);
            base.InteractableUnset(interactable);
        }

        protected override void InteractableSelected(TInteractable interactable)
        {
            base.InteractableSelected(interactable);
            GeneratePointerEvent(PointerEvent.Select, interactable);
        }

        protected override void InteractableUnselected(TInteractable interactable)
        {
            GeneratePointerEvent(PointerEvent.Unselect, interactable);
            base.InteractableUnselected(interactable);
        }

        protected override void DoPostprocess()
        {
            base.DoPostprocess();
            if (_interactable != null)
            {
                GeneratePointerEvent(PointerEvent.Move, _interactable);
            }
        }

        protected abstract Pose ComputePointerPose();
    }
}
