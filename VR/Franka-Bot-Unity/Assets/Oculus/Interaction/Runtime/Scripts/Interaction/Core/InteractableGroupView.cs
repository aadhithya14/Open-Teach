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
using UnityEngine.Assertions;

namespace Oculus.Interaction
{
    /// <summary>
    /// This class implements higher level logic to forward the highest IInteractable
    /// state of any of the interactables in its list
    /// </summary>
    public class InteractableGroupView : MonoBehaviour, IInteractableView
    {
        [SerializeField, Interface(typeof(IInteractable))]
        private List<MonoBehaviour> _interactables;
        private List<IInteractable> Interactables;

        public int InteractorsCount
        {
            get
            {
                int count = 0;
                foreach (IInteractable interactable in Interactables)
                {
                    count += interactable.InteractorsCount;
                }

                return count;
            }
        }

        public int SelectingInteractorsCount
        {
            get
            {
                int count = 0;
                foreach (IInteractable interactable in Interactables)
                {
                    count += interactable.SelectingInteractorsCount;
                }

                return count;
            }
        }

        public int MaxInteractors
        {
            get
            {
                int max = 0;
                foreach (IInteractable interactable in Interactables)
                {
                    max = Mathf.Max(interactable.MaxInteractors, max);
                }

                return max;
            }
        }

        public int MaxSelectingInteractors
        {
            get
            {
                int max = 0;
                foreach (IInteractable interactable in Interactables)
                {
                    max = Mathf.Max(interactable.MaxSelectingInteractors, max);
                }

                return max;
            }
        }

        public event Action WhenInteractorsCountUpdated = delegate { };
        public event Action WhenSelectingInteractorsCountUpdated = delegate { };

        public event Action<InteractableStateChangeArgs> WhenStateChanged = delegate { };

        private InteractableState _state = InteractableState.Normal;
        public InteractableState State
        {
            get
            {
                return _state;
            }
            set
            {
                if (_state == value) return;
                InteractableState previousState = _state;
                _state = value;
                WhenStateChanged(new InteractableStateChangeArgs { PreviousState = previousState, NewState = _state });
            }
        }

        private void UpdateState()
        {
            if (SelectingInteractorsCount > 0)
            {
                State = InteractableState.Select;
                return;
            }
            if (InteractorsCount > 0)
            {
                State = InteractableState.Hover;
                return;
            }
            State = InteractableState.Normal;
        }

        protected virtual void Awake()
        {
            Interactables = _interactables.ConvertAll(mono => mono as IInteractable);
        }

        protected bool _started = false;

        protected virtual void Start()
        {
            this.BeginStart(ref _started);
            foreach (IInteractable interactable in Interactables)
            {
                Assert.IsNotNull(interactable);
            }
            this.EndStart(ref _started);
        }

        protected virtual void OnEnable()
        {
            if (_started)
            {
                foreach (IInteractable interactable in Interactables)
                {
                    interactable.WhenStateChanged += HandleStateChange;
                }
            }
        }

        protected virtual void OnDisable()
        {
            if (_started)
            {
                foreach (IInteractable interactable in Interactables)
                {
                    interactable.WhenStateChanged -= HandleStateChange;
                }
            }
        }

        private void HandleStateChange(InteractableStateChangeArgs args)
        {
            UpdateState();
        }

        #region Inject

        public void InjectAllInteractableGroupView(List<IInteractable> interactables)
        {
            InjectInteractables(interactables);
        }

        public void InjectInteractables(List<IInteractable> interactables)
        {
            Interactables = interactables;
            _interactables =
                Interactables.ConvertAll(interactable => interactable as MonoBehaviour);
        }
        #endregion
    }
}
