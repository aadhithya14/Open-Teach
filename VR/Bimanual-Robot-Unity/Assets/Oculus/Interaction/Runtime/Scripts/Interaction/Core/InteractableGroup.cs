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
using UnityEngine.Serialization;

namespace Oculus.Interaction
{
    /// <summary>
    /// This class implements higher level logic to share the number of max
    /// max interactors acting on this group of interactors at a time.
    /// </summary>
    public class InteractableGroup : MonoBehaviour
    {
        [SerializeField, Interface(typeof(IInteractable))]
        private List<MonoBehaviour> _interactables;
        private List<IInteractable> Interactables;
        private List<InteractableLimits> _limits;

        private struct InteractableLimits
        {
            public int MaxInteractors;
            public int MaxSelectingInteractors;
        }

        [SerializeField]
        private int _maxInteractors;

        [SerializeField]
        private int _maxSelectingInteractors;

        private int _interactors;
        private int _selectInteractors;

        protected bool _started = false;

        protected virtual void Awake()
        {
            Interactables = _interactables.ConvertAll(mono => mono as IInteractable);
        }

        protected virtual void Start()
        {
            this.BeginStart(ref _started);
            foreach (IInteractable interactable in Interactables)
            {
                Assert.IsNotNull(interactable);
            }

            Assert.IsTrue(_interactables != null && _interactables.Count > 0);

            _limits = new List<InteractableLimits>();
            foreach (IInteractable interactable in Interactables)
            {
                _limits.Add(new InteractableLimits()
                {
                    MaxInteractors = interactable.MaxInteractors,
                    MaxSelectingInteractors = interactable.MaxSelectingInteractors
                });
            }
            this.EndStart(ref _started);
        }

        protected virtual void OnEnable()
        {
            if (_started)
            {
                foreach (IInteractable interactable in Interactables)
                {
                    interactable.WhenInteractorsCountUpdated += CountWhenInteractors;
                    interactable.WhenSelectingInteractorsCountUpdated += CountWhenSelectingInteractors;
                }

                CountWhenInteractors();
                CountWhenSelectingInteractors();
            }
        }

        protected virtual void OnDisable()
        {
            if (_started)
            {
                foreach (IInteractable interactable in Interactables)
                {
                    interactable.WhenInteractorsCountUpdated -= CountWhenInteractors;
                    interactable.WhenSelectingInteractorsCountUpdated -= CountWhenSelectingInteractors;
                }

                CountWhenInteractors();
                CountWhenSelectingInteractors();
            }
        }

        private void CountWhenInteractors()
        {
            _interactors = 0;
            foreach (IInteractable interactable in Interactables)
            {
                _interactors += interactable.InteractorsCount;
            }

            UpdateMaxInteractors();
        }

        private void UpdateMaxInteractors()
        {
            if (_maxInteractors == -1) return;
            int remainingInteractors = Mathf.Max(0, _maxInteractors - _interactors);
            for (int i = 0; i < Interactables.Count; i++)
            {
                Interactables[i].MaxInteractors = (_limits[i].MaxInteractors == -1
                    ? remainingInteractors
                    : Mathf.Max(0, _limits[i].MaxInteractors - _interactors)) +
                                                  Interactables[i].InteractorsCount;
            }
        }

        private void CountWhenSelectingInteractors()
        {
            _selectInteractors = 0;
            foreach (IInteractable interactable in Interactables)
            {
                _selectInteractors += interactable.SelectingInteractorsCount;
            }

            UpdateMaxActive();
        }

        private void UpdateMaxActive()
        {
            if (_maxSelectingInteractors == -1) return;
            int remainingActive = Mathf.Max(0, _maxSelectingInteractors - _selectInteractors);
            for (int i = 0; i < Interactables.Count; i++)
            {
                Interactables[i].MaxSelectingInteractors = (_limits[i].MaxSelectingInteractors == -1
                    ? remainingActive
                    : Mathf.Max(0, _limits[i].MaxSelectingInteractors - _selectInteractors)) +
                                                        Interactables[i].SelectingInteractorsCount;
            }
        }

        #region Inject

        public void InjectAllInteractableGroup(List<IInteractable> interactables)
        {
            InjectInteractables(interactables);
        }

        public void InjectInteractables(List<IInteractable> interactables)
        {
            Interactables = interactables;
            _interactables =
                interactables.ConvertAll(interactable => interactable as MonoBehaviour);
        }

        #endregion
    }
}
