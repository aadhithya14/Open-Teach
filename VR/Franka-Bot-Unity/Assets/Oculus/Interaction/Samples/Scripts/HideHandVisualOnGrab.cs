/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using Oculus.Interaction.HandGrab;
using System;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Serialization;

namespace Oculus.Interaction.Samples
{
    public class HideHandVisualOnGrab : MonoBehaviour
    {
        [SerializeField]
        private HandGrabInteractor _handGrabInteractor;

        [SerializeField]
        private HandVisual _handVisual;

        protected virtual void Start()
        {
            Assert.IsNotNull(_handVisual);
        }

        protected virtual void Update()
        {
            GameObject shouldHideHandComponent = null;

            if (_handGrabInteractor.State == InteractorState.Select)
            {
                shouldHideHandComponent = _handGrabInteractor.SelectedInteractable?.gameObject;
            }

            if (shouldHideHandComponent)
            {
                if (shouldHideHandComponent.TryGetComponent(out ShouldHideHandOnGrab component))
                {
                    _handVisual.ForceOffVisibility = true;
                }
            }
            else
            {
                _handVisual.ForceOffVisibility = false;
            }
        }

        #region Inject

        public void InjectAll(HandGrabInteractor handGrabInteractor,
             HandVisual handVisual)
        {
            InjectHandGrabInteractor(handGrabInteractor);
            InjectHandVisual(handVisual);
        }
        private void InjectHandGrabInteractor(HandGrabInteractor handGrabInteractor)
        {
            _handGrabInteractor = handGrabInteractor;
        }

        private void InjectHandVisual(HandVisual handVisual)
        {
            _handVisual = handVisual;
        }


        #endregion
    }
}
