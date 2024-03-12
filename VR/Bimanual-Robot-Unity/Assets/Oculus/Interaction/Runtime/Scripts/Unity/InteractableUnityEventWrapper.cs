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
using UnityEngine.Assertions;
using UnityEngine.Events;

namespace Oculus.Interaction
{
    /// <summary>
    /// This component makes it possible to connect Interactables in the
    /// inspector to Unity Events that are broadcast on state changes.
    /// </summary>
    public class InteractableUnityEventWrapper : MonoBehaviour
    {
        [SerializeField, Interface(typeof(IInteractableView))]
        private MonoBehaviour _interactableView;
        private IInteractableView InteractableView;

        [SerializeField]
        private UnityEvent _whenHover;
        [SerializeField]
        private UnityEvent _whenUnhover;
        [SerializeField]
        private UnityEvent _whenSelect;
        [SerializeField]
        private UnityEvent _whenUnselect;

        [SerializeField]
        private UnityEvent _whenInteractorsCountUpdated;
        [SerializeField]
        private UnityEvent _whenSelectingInteractorsCountUpdated;

        #region Properties

        public UnityEvent WhenHover => _whenHover;
        public UnityEvent WhenUnhover => _whenUnhover;
        public UnityEvent WhenSelect => _whenSelect;
        public UnityEvent WhenUnselect => _whenUnselect;
        public UnityEvent WhenInteractorsCountUpdated => _whenInteractorsCountUpdated;
        public UnityEvent WhenSelectingInteractorsCountUpdated => _whenSelectingInteractorsCountUpdated;

        #endregion

        protected bool _started = false;

        protected virtual void Awake()
        {
            InteractableView = _interactableView as IInteractableView;
        }

        protected virtual void Start()
        {
            this.BeginStart(ref _started);
            Assert.IsNotNull(InteractableView);
            this.EndStart(ref _started);
        }

        protected virtual void OnEnable()
        {
            if (_started)
            {
                InteractableView.WhenStateChanged += HandleStateChanged;
                InteractableView.WhenInteractorsCountUpdated += HandleInteractorsCountUpdated;
                InteractableView.WhenSelectingInteractorsCountUpdated += HandleSelectingInteractorsCountUpdated;
            }
        }

        protected virtual void OnDisable()
        {
            if (_started)
            {
                InteractableView.WhenStateChanged -= HandleStateChanged;
                InteractableView.WhenInteractorsCountUpdated -= HandleInteractorsCountUpdated;
                InteractableView.WhenSelectingInteractorsCountUpdated -= HandleSelectingInteractorsCountUpdated;
            }
        }

        private void HandleStateChanged(InteractableStateChangeArgs args)
        {
            switch (args.NewState)
            {
                case InteractableState.Normal:
                    if (args.PreviousState == InteractableState.Hover)
                    {
                        _whenUnhover.Invoke();
                    }

                    break;
                case InteractableState.Hover:
                    if (args.PreviousState == InteractableState.Normal)
                    {
                        _whenHover.Invoke();
                    }
                    else if (args.PreviousState == InteractableState.Select)
                    {
                        _whenUnselect.Invoke();
                    }

                    break;
                case InteractableState.Select:
                    if (args.PreviousState == InteractableState.Hover)
                    {
                        _whenSelect.Invoke();
                    }

                    break;
            }
        }

        private void HandleInteractorsCountUpdated()
        {
            _whenInteractorsCountUpdated.Invoke();
        }

        private void HandleSelectingInteractorsCountUpdated()
        {
            _whenSelectingInteractorsCountUpdated.Invoke();
        }

        #region Inject

        public void InjectAllInteractableUnityEventWrapper(IInteractableView interactableView)
        {
            InjectInteractableView(interactableView);
        }

        public void InjectInteractableView(IInteractableView interactableView)
        {
            _interactableView = interactableView as MonoBehaviour;
            InteractableView = interactableView;
        }

        #endregion
    }
}
