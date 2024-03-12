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

namespace Oculus.Interaction
{
    /// <summary>
    /// A registry that houses a set of concrete Interactables.
    /// </summary>
    public class InteractableRegistry<TInteractor, TInteractable>
                                     where TInteractor : Interactor<TInteractor, TInteractable>
                                     where TInteractable : Interactable<TInteractor, TInteractable>
    {
        private static List<TInteractable> _interactables;
        private List<TInteractable> _interactableEnumeratorList;

        public InteractableRegistry()
        {
            _interactables = new List<TInteractable>();
            _interactableEnumeratorList = new List<TInteractable>();
        }

        public virtual void Register(TInteractable interactable) => _interactables.Add(interactable);
        public virtual void Unregister(TInteractable interactable) => _interactables.Remove(interactable);

        protected IEnumerable<TInteractable> PruneInteractables(IEnumerable<TInteractable> interactables,
                                                            TInteractor interactor)
        {
            int interactableCount = 0;
            foreach (TInteractable interactable in interactables)
            {
                if (!interactor.IsFilterPassedBy(interactable))
                {
                    continue;
                }

                if (!interactable.CanBeSelectedBy(interactor))
                {
                    continue;
                }

                if (interactableCount == _interactableEnumeratorList.Count)
                {
                    _interactableEnumeratorList.Add(interactable);
                }
                else
                {
                    _interactableEnumeratorList[interactableCount] = interactable;
                }
                interactableCount++;
            }

            return GetRange(_interactableEnumeratorList, 0, interactableCount);
        }

        public virtual IEnumerable<TInteractable> List(TInteractor interactor)
        {
            return PruneInteractables(_interactables, interactor);
        }

        public virtual IEnumerable<TInteractable> List()
        {
            return _interactables;
        }

        private IEnumerable<T> GetRange<T>(IEnumerable<T> source, int start, int end)
        {
            using (IEnumerator<T> e = source.GetEnumerator())
            {
                int i = 0;
                while (i < start && e.MoveNext()) { i++; }
                while (i < end && e.MoveNext())
                {
                    yield return e.Current;
                    i++;
                }
            }
        }
    }
}
