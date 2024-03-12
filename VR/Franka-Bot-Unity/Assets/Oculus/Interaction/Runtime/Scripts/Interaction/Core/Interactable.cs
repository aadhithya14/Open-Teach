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
using System.Collections.Generic;
using UnityEngine;

namespace Oculus.Interaction
{
    /// <summary>
    /// Interactable provides a base template for any kind of interactable object.
    /// An Interactable can have Hover and HandleSelected Interactor(s) acting on it.
    /// Concrete Interactables can define whether they have a One-to-One or
    /// One-to-Many relationship with their associated concrete Interactors.
    /// </summary>
    public abstract class Interactable<TInteractor, TInteractable> : MonoBehaviour, IInteractable
                                        where TInteractor : Interactor<TInteractor, TInteractable>
                                        where TInteractable : Interactable<TInteractor, TInteractable>
    {
        /// <summary>
        /// The max Interactors and max selecting Interactors that this Interactable can
        /// have acting on it.
        /// -1 signifies NO limit (can have any number of Interactors)
        /// </summary>
        [SerializeField]
        private int _maxInteractors = -1;

        [SerializeField]
        private int _maxSelectingInteractors = -1;

        #region Properties
        public int MaxInteractors
        {
            get
            {
                return _maxInteractors;
            }
            set
            {
                _maxInteractors = value;
            }
        }

        public int MaxSelectingInteractors
        {
            get
            {
                return _maxSelectingInteractors;
            }
            set
            {
                _maxSelectingInteractors = value;
            }
        }
        #endregion

        private HashSet<TInteractor> _interactors = new HashSet<TInteractor>();
        private HashSet<TInteractor> _selectingInteractors = new HashSet<TInteractor>();

        public event Action WhenInteractorsCountUpdated = delegate { };
        public event Action WhenSelectingInteractorsCountUpdated = delegate { };

        private InteractableState _state = InteractableState.Disabled;
        public event Action<InteractableStateChangeArgs> WhenStateChanged = delegate { };

        private MultiAction<TInteractor> _whenInteractorAdded = new MultiAction<TInteractor>();
        private MultiAction<TInteractor> _whenInteractorRemoved = new MultiAction<TInteractor>();
        private MultiAction<TInteractor> _whenSelectingInteractorAdded = new MultiAction<TInteractor>();
        private MultiAction<TInteractor> _whenSelectingInteractorRemoved = new MultiAction<TInteractor>();
        public MAction<TInteractor> WhenInteractorAdded => _whenInteractorAdded;
        public MAction<TInteractor> WhenInteractorRemoved => _whenInteractorRemoved;
        public MAction<TInteractor> WhenSelectingInteractorAdded => _whenSelectingInteractorAdded;
        public MAction<TInteractor> WhenSelectingInteractorRemoved => _whenSelectingInteractorRemoved;

        public InteractableState State
        {
            get
            {
                return _state;
            }
            private set
            {
                if (_state == value) return;
                InteractableState previousState = _state;
                _state = value;
                WhenStateChanged(new InteractableStateChangeArgs
                {
                    PreviousState = previousState,
                    NewState = _state
                });
            }
        }

        private static InteractableRegistry<TInteractor, TInteractable> _registry =
                                        new InteractableRegistry<TInteractor, TInteractable>();

        public static InteractableRegistry<TInteractor, TInteractable> Registry => _registry;

        protected virtual void InteractorAdded(TInteractor interactor)
        {
            _whenInteractorAdded.Invoke(interactor);
        }
        protected virtual void InteractorRemoved(TInteractor interactor)
        {
            _whenInteractorRemoved.Invoke(interactor);
        }

        protected virtual void SelectingInteractorAdded(TInteractor interactor)
        {
            _whenSelectingInteractorAdded.Invoke(interactor);
        }
        protected virtual void SelectingInteractorRemoved(TInteractor interactor)
        {
            _whenSelectingInteractorRemoved.Invoke(interactor);
        }

        public int InteractorsCount => _interactors.Count;

        public int SelectingInteractorsCount => _selectingInteractors.Count;

        public IEnumerable<TInteractor> Interactors => _interactors;

        public IEnumerable<TInteractor> SelectingInteractors => _selectingInteractors;

        public void AddInteractor(TInteractor interactor)
        {
            _interactors.Add(interactor);
            WhenInteractorsCountUpdated();
            InteractorAdded(interactor);
            UpdateInteractableState();
        }

        public void RemoveInteractor(TInteractor interactor)
        {
            if (!_interactors.Remove(interactor))
            {
                return;
            }
            interactor.InteractableChangesUpdate();
            WhenInteractorsCountUpdated();
            InteractorRemoved(interactor);
            UpdateInteractableState();
        }

        public void AddSelectingInteractor(TInteractor interactor)
        {
            _selectingInteractors.Add(interactor);
            WhenSelectingInteractorsCountUpdated();
            SelectingInteractorAdded(interactor);
            UpdateInteractableState();
        }

        public void RemoveSelectingInteractor(TInteractor interactor)
        {
            if (!_selectingInteractors.Remove(interactor))
            {
                return;
            }
            interactor.InteractableChangesUpdate();
            WhenSelectingInteractorsCountUpdated();
            SelectingInteractorRemoved(interactor);
            UpdateInteractableState();
        }

        private void UpdateInteractableState()
        {
            if (State == InteractableState.Disabled) return;

            if (SelectingInteractorsCount > 0)
            {
                State = InteractableState.Select;
            }
            else if (InteractorsCount > 0)
            {
                State = InteractableState.Hover;
            }
            else
            {
                State = InteractableState.Normal;
            }
        }

        public bool CanBeSelectedBy(TInteractor interactor)
        {
            if (State == InteractableState.Disabled)
            {
                return false;
            }

            if (MaxSelectingInteractors >= 0 &&
                SelectingInteractorsCount == MaxSelectingInteractors)
            {
                return false;
            }

            if (MaxInteractors >= 0 &&
                InteractorsCount == MaxInteractors &&
                !_interactors.Contains(interactor))
            {
                return false;
            }

            return true;
        }

        public bool HasInteractor(TInteractor interactor)
        {
            return _interactors.Contains(interactor);
        }

        public bool HasSelectingInteractor(TInteractor interactor)
        {
            return _selectingInteractors.Contains(interactor);
        }

        public void Enable()
        {
            if (State != InteractableState.Disabled)
            {
                return;
            }
            _registry.Register((TInteractable)this);
            State = InteractableState.Normal;
        }

        public void Disable()
        {
            if (State == InteractableState.Disabled)
            {
                return;
            }

            List<TInteractor> selectingInteractorsCopy = new List<TInteractor>(_selectingInteractors);
            foreach (TInteractor selectingInteractor in selectingInteractorsCopy)
            {
                RemoveSelectingInteractor(selectingInteractor);
            }

            List<TInteractor> interactorsCopy = new List<TInteractor>(_interactors);
            foreach (TInteractor interactor in interactorsCopy)
            {
                RemoveInteractor(interactor);
            }

            State = InteractableState.Disabled;
            _registry.Unregister((TInteractable)this);
        }

        public void RemoveInteractorById(int id)
        {
            TInteractor foundInteractor = null;
            foreach (TInteractor selectingInteractor in _selectingInteractors)
            {
                if (selectingInteractor.Identifier == id)
                {
                    foundInteractor = selectingInteractor;
                    break;
                }
            }

            if (foundInteractor != null)
            {
                RemoveSelectingInteractor(foundInteractor);
            }

            foundInteractor = null;

            foreach (TInteractor interactor in _interactors)
            {
                if (interactor.Identifier == id)
                {
                    foundInteractor = interactor;
                    break;
                }
            }

            if (foundInteractor == null)
            {
                return;
            }

            RemoveInteractor(foundInteractor);
        }

        protected virtual void OnEnable()
        {
            Enable();
        }

        protected virtual void OnDisable()
        {
            Disable();
        }

        protected virtual void SetRegistry(InteractableRegistry<TInteractor, TInteractable> registry)
        {
            if (registry == _registry) return;

            IEnumerable<TInteractable> interactables = _registry.List();
            foreach (TInteractable interactable in interactables)
            {
                registry.Register(interactable);
                _registry.Unregister(interactable);
            }
            _registry = registry;
        }
    }
}
